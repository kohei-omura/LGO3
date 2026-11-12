/* ════════════════════════════════════════════════════════════
   端末内生成を Worker で実行する（iPhone の強制終了対策）
     ・iPhone の Safari は「JSのメモリ＋GPUのメモリ」の合計に約1.9GBの上限がある（実機テストで推定）。
       モデルファイル（数百MB）を丸ごとメモリへ読むと、GPUへ送った後も写しが残って上限を超えていた
     ・ここでは OPFS の同期ハンドルで「重み1個ずつ」読み、GPU へ送ったらすぐ使い回す
       （onnxruntime の外部データ読み出しを readSlice に対応させてある: ort-wasm-*.mjs の LGO3 パッチ）
     ・1枚生成するごとに Worker ごと捨てるので、GPU も含めメモリが確実に返る
     ・生成の間は GPU 接続を保つ（LocalGenCore.keepAlive。部品の切り替えで作り直すと iPhone で落ちる）
   メッセージ:
     → { cmd:'probe', ep }
     ← { type:'probe', gpu, adapter, f16, sync }
     → { cmd:'generate', tier, root, dtype, ortBase, mjsQuery, graphBase, ep, epOptions, epFor, stream, keepAlive, opts }
     ← { type:'progress', ph, f } / { type:'done', r } / { type:'error', message, name } / { type:'log', text }（debug 時）
   ════════════════════════════════════════════════════════════ */
'use strict';

async function probe(m) {
  const out = { type: 'probe', gpu: !!(self.navigator && navigator.gpu), adapter: false, f16: false,
                sync: typeof self.FileSystemSyncAccessHandle !== 'undefined' };
  if (out.gpu && m.ep === 'webgpu') {
    try {
      const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      out.adapter = !!ad; out.f16 = !!(ad && ad.features.has('shader-f16'));
    } catch (e) { out.error = String(e && e.message || e); }
  }
  return out;
}

let _ort = false;
function loadOrt(m) {
  if (_ort) return;
  importScripts(m.ortBase + 'ort.webgpu.min.js', m.coreUrl);
  ort.env.wasm.wasmPaths = { mjs: m.ortBase + 'ort-wasm-simd-threaded.asyncify.mjs' + (m.mjsQuery || ''),
                             wasm: m.ortBase + 'ort-wasm-simd-threaded.asyncify.wasm' };
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  try { ort.env.webgpu.powerPreference = 'high-performance'; } catch (e) {}
  _ort = true;
}

async function dirOf(path) {
  let d = await navigator.storage.getDirectory();
  for (const p of path.split('/').filter(Boolean)) d = await d.getDirectoryHandle(p);
  return d;
}

// 重みの読み出し係。onnxruntime が重み1個ごとに readSlice(位置, 長さ) を呼ぶ。
// 返した領域は呼び出し側がすぐ（同期的に）GPU / wasm へ写すので、作業用の1枚を使い回してよい
let scratch = new Uint8Array(0);
function makeReader(h) {
  return {
    byteLength: h.getSize(),
    readSlice(off, len) {
      if (scratch.length < len) scratch = new Uint8Array(Math.max(len, 16 << 20));
      const v = scratch.subarray(0, len);
      const n = h.read(v, { at: off });
      if (n !== len) throw new Error(`重みの読み込みが不足しました（${n}/${len}）`);
      return v;
    },
  };
}

function makeEnv(m) {
  const epOf = comp => {
    const e0 = (m.epFor && m.epFor[comp]) || m.ep;
    return e0 === 'webgpu' && m.epOptions ? Object.assign({ name: 'webgpu' }, m.epOptions) : e0;
  };
  // GPU 接続を生成の間ずっと保つ（部品の切り替えで作り直させない）
  const ka = m.ep === 'webgpu' && m.keepAlive !== false ? LocalGenCore.keepAlive(ort, epOf('unet')) : null;
  return {
    ort, dtype: m.dtype,
    begin: ka && ka.begin, end: ka && ka.end,
    loadModel: async comp => {
      const name = comp + '.onnx';
      const fh = await (await dirOf(m.root + '/' + m.tier)).getFileHandle(name);
      const g = await fetch(m.graphBase + name);
      if (!g.ok) throw new Error('設計図ファイルを読み込めません: ' + comp);
      const model = new Uint8Array(await g.arrayBuffer());
      if (!m.stream) {
        const f = await fh.getFile();
        return { model, externalData: [{ path: name, data: new Uint8Array(await f.arrayBuffer()) }] };
      }
      const h = await fh.createSyncAccessHandle();
      (self.LGO_EXT_READERS = self.LGO_EXT_READERS || {})[name] = makeReader(h);
      return { model, externalData: [{ path: name, data: new Uint8Array(1) }],   // 実体は読み出し係が渡す
               done: () => { delete self.LGO_EXT_READERS[name]; try { h.close(); } catch (e) {} } };
    },
    loadTokenizer: async () => {
      const c = await dirOf(m.root + '/common');
      const read = async n => (await (await c.getFileHandle(n)).getFile()).text();
      return { vocab: JSON.parse(await read('vocab.json')), merges: await read('merges.txt') };
    },
    sessionOptions: (comp, d) => {
      const ov = comp === 'text_encoder' ? { batch_size: d.batch, sequence_length: 77 }
               : comp === 'unet'         ? { batch_size: d.batch, height: d.h, width: d.w, sequence_length: 77 }
               :                           { batch_size: d.batch, latent_height: d.h, latent_width: d.w };
      return { executionProviders: [epOf(comp)], graphOptimizationLevel: 'all', freeDimensionOverrides: ov, logSeverityLevel: 3 };
    },
  };
}

// 診断用: GPU 接続の作成・破棄とエラーを記録して知らせる（m.debug の時だけ）
function traceGpu() {
  if (!self.GPUAdapter || GPUAdapter.prototype.__lgoTraced) return;
  GPUAdapter.prototype.__lgoTraced = true;
  const log = text => self.postMessage({ type: 'log', text });
  const orig = GPUAdapter.prototype.requestDevice;
  let n = 0;
  GPUAdapter.prototype.requestDevice = async function (...a) {
    const d = await orig.apply(this, a), id = ++n;
    log(`GPU接続#${id} 作成`);
    const od = d.destroy.bind(d);
    d.destroy = () => { log(`GPU接続#${id} 破棄`); return od(); };
    d.addEventListener('uncapturederror', ev => log(`GPUエラー#${id}: ` + String(ev.error && ev.error.message).slice(0, 150)));
    d.lost.then(i => { if (i && i.reason !== 'destroyed') log(`GPU接続#${id} 喪失: ` + (i.message || i.reason)); });
    return d;
  };
}

self.onmessage = async e => {
  const m = e.data;
  try {
    if (m.cmd === 'probe') { self.postMessage(await probe(m)); return; }
    if (m.cmd !== 'generate') return;
    if (m.debug) traceGpu();
    if (m.uploadMode) self.LGO_ORT_UPLOAD = m.uploadMode;
    loadOrt(m);
    const r = await LocalGenCore.generate(makeEnv(m), Object.assign({}, m.opts, {
      onProgress: (ph, f) => self.postMessage({ type: 'progress', ph, f }),
    }));
    self.postMessage({ type: 'done', r }, [r.rgba.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err), name: err && err.name });
  }
};
