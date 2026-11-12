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
const WORKER_VERSION = '2026-11-13d';

async function probe(m) {
  const out = { type: 'probe', gpu: !!(self.navigator && navigator.gpu), adapter: false, f16: false,
                sync: typeof self.FileSystemSyncAccessHandle !== 'undefined',
                version: WORKER_VERSION, gpuAdapterGlobal: typeof self.GPUAdapter !== 'undefined' };
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
  importScripts(m.ortBase + 'ort.webgpu.min.js' + (m.assetQuery || ''), m.coreUrl);
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
      const g = await fetch(m.graphBase + name + (m.assetQuery || ''));
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
//   GPUAdapter が全体に公開されていない環境もあるため、navigator.gpu.requestAdapter の戻り値を直接くるむ
let _traced = false;
function traceGpu() {
  if (_traced || !(self.navigator && navigator.gpu)) return;
  _traced = true;
  const log = text => self.postMessage({ type: 'log', text });
  const g = navigator.gpu, origAd = g.requestAdapter.bind(g);
  let n = 0;
  g.requestAdapter = async (...a) => {
    const ad = await origAd(...a);
    if (!ad) { log('GPU 取得失敗'); return ad; }
    const origDev = ad.requestDevice.bind(ad);
    ad.requestDevice = async (...b) => {
      const d = await origDev(...b), id = ++n;
      log(`GPU接続#${id} 作成`);
      const od = d.destroy.bind(d);
      d.destroy = () => { log(`GPU接続#${id} 破棄`); return od(); };
      d.addEventListener('uncapturederror', ev => log(`GPUエラー#${id}: ` + String(ev.error && ev.error.message).slice(0, 150)));
      d.lost.then(i => { if (i && i.reason !== 'destroyed') log(`GPU接続#${id} 喪失: ` + (i.message || i.reason)); });
      return d;
    };
    return ad;
  };
}

// ── 診断（diag-gen.html 用）: 部品を選んで動かし、入出力の値を調べる ──
const statsOf = a => { let mx = 0, bad = 0, sum = 0; for (let i = 0; i < a.length; i++) { const v = a[i]; if (!Number.isFinite(v)) { bad++; continue; } const x = Math.abs(v); if (x > mx) mx = x; sum += v; }
                       return { 最大絶対値: +mx.toFixed(2), 異常値: bad, 平均: +(sum / a.length).toFixed(4) }; };
async function fileDir(path) { let d = await navigator.storage.getDirectory(); for (const p of path.split('/').filter(Boolean)) d = await d.getDirectoryHandle(p, { create: true }); return d; }
async function saveBin(path, name, u8) { const h = await (await (await fileDir(path)).getFileHandle(name, { create: true })).createSyncAccessHandle(); try { h.truncate(0); h.write(u8, { at: 0 }); h.flush(); } finally { h.close(); } }
async function loadBin(path, name) { const h = await (await (await fileDir(path)).getFileHandle(name)).createSyncAccessHandle(); try { const u8 = new Uint8Array(h.getSize()); h.read(u8, { at: 0 }); return u8; } finally { h.close(); } }
async function diag(m) {
  loadOrt(m);
  const C = LocalGenCore, env = makeEnv(m), p = m.plan, out = {}, L = 77, HS = 77 * 768;
  const log = text => self.postMessage({ type: 'log', text });
  const prog = ph => log(ph);
  if (env.begin) { await env.begin(); log('つなぎ 作成'); }
  try {
    let hid = null;
    if (p.text) {
      const t = await env.loadTokenizer(), tok = new C.CLIPTokenizer(t.vocab, t.merges);
      const neg = tok.encode(p.negative || ''), pos = tok.encode(p.prompt || '');
      const ids = new BigInt64Array(2 * L);
      neg.ids.forEach((v, i) => { ids[i] = BigInt(v); }); pos.ids.forEach((v, i) => { ids[L + i] = BigInt(v); });
      hid = await C._withSession(env, 'text_encoder', { batch: 2 }, async s => C.toF32((await s.run({ input_ids: new ort.Tensor('int64', ids, [2, L]) })).last_hidden_state.data), prog);
      out.文章部品の出力 = statsOf(hid);
      log('文章部品の出力 ' + JSON.stringify(out.文章部品の出力));
      if (p.saveEmb) { await saveBin(m.root + '/diag', 'emb.bin', new Uint8Array(hid.buffer.slice(0))); log('文章部品の出力を保存'); }
    }
    if (p.unet) {
      const u = p.unet, lh = u.h / 8, lw = u.w / 8, n = 4 * lh * lw, dt = m.dtype;
      if (u.emb === 'saved') { const b = await loadBin(m.root + '/diag', 'emb.bin'); hid = new Float32Array(b.buffer, 0, b.byteLength / 4); log('保存した文章部品の出力を使用 ' + JSON.stringify(statsOf(hid))); }
      if (u.emb === 'synthetic' || !hid) { hid = new Float32Array(2 * HS); for (let i = 0; i < hid.length; i++) hid[i] = Math.sin(i * 0.37) * 0.5; }
      let sample;
      if (u.sample === 'gauss') sample = C.gaussian(n, 1234);
      else { sample = new Float32Array(n); for (let i = 0; i < n; i++) sample[i] = Math.sin(i * 0.37) * 0.5; }
      const ctx = [0, 1].map(k => C._fromF32(ort, dt, hid.subarray(k * HS, (k + 1) * HS), [1, L, 768]));
      await C._withSession(env, 'unet', { batch: 1, h: lh, w: lw }, async s => {
        for (let r = 0; r < (u.runs || 2); r++) {
          log(`${r + 1}回目の実行中…`);
          const ts = C._fromF32(ort, dt, new Float32Array([u.timestep || 999]), []);
          const o = C.toF32((await s.run({ sample: C._fromF32(ort, dt, sample, [1, 4, lh, lw]), timestep: ts, encoder_hidden_states: ctx[r % 2] })).out_sample.data);
          out[`描画${r + 1}回目の出力`] = statsOf(o);
          log(`${r + 1}回目 完了 ` + JSON.stringify(out[`描画${r + 1}回目の出力`]));
        }
      }, prog);
    }
    return out;
  } finally { if (env.end) { await env.end(); log('つなぎ 解放'); } }
}

self.onmessage = async e => {
  const m = e.data;
  try {
    if (m.cmd === 'probe') { self.postMessage(await probe(m)); return; }
    if (m.debug) traceGpu();
    if (m.cmd === 'diag') {
      if (m.uploadMode) self.LGO_ORT_UPLOAD = m.uploadMode;
      loadOrt(m);
      self.postMessage({ type: 'log', text: `Worker ${WORKER_VERSION} / 計算部品 ${LocalGenCore.VERSION || '古い版'}` });
      if (m.expectVersion && LocalGenCore.VERSION !== m.expectVersion) throw new Error(`古い計算部品が読み込まれました（${LocalGenCore.VERSION || '不明'}）。アプリを開き直してください`);
      const out = await diag(m);
      self.postMessage({ type: 'done', r: { diag: out } });
      return;
    }
    if (m.cmd !== 'generate') return;
    if (m.uploadMode) self.LGO_ORT_UPLOAD = m.uploadMode;
    loadOrt(m);
    if (m.debug) self.postMessage({ type: 'log', text: `Worker ${WORKER_VERSION} / 計算部品 ${LocalGenCore.VERSION}` });
    const r = await LocalGenCore.generate(makeEnv(m), Object.assign({}, m.opts, {
      onProgress: (ph, f) => self.postMessage({ type: 'progress', ph, f }),
    }));
    self.postMessage({ type: 'done', r }, [r.rgba.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err), name: err && err.name });
  }
};
