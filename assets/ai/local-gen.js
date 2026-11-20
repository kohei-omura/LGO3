/* ════════════════════════════════════════════════════════════
   端末内（オフライン）画像生成 ── ブラウザ側
     モデル: segmind/tiny-sd を ONNX 化した cursedhelm/aderpy-deepdreamer-onnx
             （CreativeML Open RAIL-M・Hugging Face から初回だけ取得し端末に保存）
     実行  : onnxruntime-web（WebGPU）。部品を1つずつ読み込み、使い終わったら解放する。
     iPhone 15 Pro の実機診断: 1ページ約1.9GBで強制終了 → 標準(fp16 約1.0GB)まで可。
   公開 API（window.LOCAL_GEN）
     supported()            … この端末で動く見込みがあるか（WebGPU + 半精度計算）
     status()               … { tiers:{M:{ready,bytes}, S:{…}}, readyTier, usage, quota }
     download(tier, onProg) … モデル取得（再開可・SHA-256 検証）。戻り値 Promise
     abortDownload()
     remove(tier)
     generate(opts)         … { blob, width, height, tier, label, msPerStep }
   ════════════════════════════════════════════════════════════ */
(function () {
  'use strict';
  const REPO = 'cursedhelm/aderpy-deepdreamer-onnx';
  const REV  = 'fa62aeaf4ee8a66f77cdc2592983210451857aae';      // 取得するリビジョンを固定（中身が変わらない）
  const ROOT_DIR = 'lgo-localgen';
  const MB = 1048576;

  const COMMON = [   // トークナイザは両モデル共通
    { name: 'vocab.json',  src: 'tiny-sd-web-fp16/tokenizer/vocab.json',  size: 1109372, sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e' },
    { name: 'merges.txt',  src: 'tiny-sd-web-fp16/tokenizer/merges.txt',  size: 573514,  sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052' },
  ];
  const TIERS = {
    M: { label: 'Tiny SD 標準（fp16）', short: '標準', dtype: 'float16', files: [
      { name: 'text_encoder.onnx', src: 'tiny-sd-web-fp16/text_encoder/model.onnx', size: 246343543, sha256: 'f9749fc65cd7652319089d48e57f40b1dc381bc77f245e49f6df1491e037f95a' },
      { name: 'unet.onnx',         src: 'tiny-sd-web-fp16/unet/model.onnx',         size: 647295610, sha256: 'ca4e66305815ec9cf0a8300cef17e5ddb4928fbcb13e428679fa353a204b8f71' },
      { name: 'vae_decoder.onnx',  src: 'tiny-sd-web-fp16/vae_decoder/model.onnx',  size: 99093735,  sha256: '3c91e9289f12f84da4987530a1b690719a229583dc8e5209cdc8601348538138' } ] },
    S: { label: 'Tiny SD 軽量（4bit）', short: '軽量', dtype: 'float16', files: [
      { name: 'text_encoder.onnx', src: 'tiny-sd-web-q4f16/text_encoder/model.onnx', size: 69631480,  sha256: 'a790756197edc41efc70bc2cfaa7f282d2d8fcf1ca3e7ccd4de58fede1ce86c8' },
      { name: 'unet.onnx',         src: 'tiny-sd-web-q4f16/unet/model.onnx',         size: 465548639, sha256: 'c994b47cb486bc157451bbd799d9293e7e7501b97689a1d2b4399f4b4b4ba61b' },
      { name: 'vae_decoder.onnx',  src: 'tiny-sd-web-q4f16/vae_decoder/model.onnx',  size: 97587114,  sha256: '11ccb1037ee17cfddadd5c2b68eb31993cbdf69c90decf164d607f9576a40b40' } ] },
  };
  // 画質モードごとの生成サイズと工程数（iPhone のメモリ上限から 448×640 が上限の目安）
  const PLANS = {
    fast:     { w: 384, h: 576, steps: 12 },
    balanced: { w: 448, h: 640, steps: 18 },
    hq:       { w: 448, h: 640, steps: 25 },
  };
  const FALLBACK_SIZE = { w: 384, h: 576 };   // メモリ不足で落ちた時に1回だけ縮めて再試行

  // テスト時のみ差し替え可能（配布元・実行方式・モデル定義）
  const ORT_Q = '?v=lgo3';   // 実行部品（LGO3 パッチ入り）を修正したら上げる。古いキャッシュを使わせない
  // 本アプリの部品の版。URL に ?v= を付けて、古い保存データ（Service Worker・ブラウザのキャッシュ）を使わせない。
  //   変えたら index.html / diag-gen.html の local-gen.js?v= と sw.js の ASSETS もそろえる
  const AV = '2026-11-17d', VQ = '?v=' + AV;
  const cfg = { base: `https://huggingface.co/${REPO}/resolve/${REV}/`, ep: 'webgpu', tiers: TIERS, common: COMMON, assetBase: 'assets/ai/', useGraphs: true,
                // GPU バッファは同じ大きさのものを使い回す（simple）。毎回作って捨てる方式（lazyRelease）だと、
                //   捨てた分が実際に返るまで時間差があり、iPhone では工程を重ねるうちに上限（2048MB）を超えた。
                //   使い回しなら2回目以降の実行で新しく作るバッファはほぼ 0（Dawn 計測。出力は同じ）
                epOptions: { storageBufferCacheMode: 'simple', uniformBufferCacheMode: 'simple' },
                // GPU への送信の扱い（Worker 内）。bypass: 待ち合わせをせず、待ち時間の記録だけ取る
                //   待ち合わせをする場合は { pending: 2, budget: 0.8, min: 2, max: 16 }
                //   （pending: 1回に送る命令の数 / budget: GPU に預ける見積もり時間(秒) / min・max: 預ける送信の数）
                //   実機テスト第12回: 待ち合わせなしだと仕上げや 448×640 で「GPU待ち」が3秒に達して GPU 側が落ちた → 待ち合わせあり
                pace: { pending: 2, budget: 0.8, min: 2, max: 16 },
                // 部品の読み込み後に、一時メモリが返るのを待つ（iPhone はページのメモリ上限が 2048MB。
                //   実機テスト第11回: 読み込み直後は約1120MB 使用 → 3秒待つと約384MB）
                //   waitMs: 待つ時間 / gc: 使い終わった GPU 関連オブジェクトの回収（GC）を促す
                //   small: 文章部品（小さい）とタイルごとの待ち時間
                settle: { waitMs: 3000, small: 400, gc: true },
                // 仕上げ（VAE）を分割する大きさ: 潜在表現の画素数がこれを超えたらタイルに分ける（1536 = 256×384 相当）
                //   17c で 1536 → 768（タイル 32×44 → 24×32 の9枚）。仕上げ部品の作業用メモリは使い終わっても GPU 側に残るため、
                //   そのあとの顔の描き直し（描画部品 448×640）と重なって落ちた（実機 17b: 3回中1回、顔の描き直しの最中）
                vaeMaxArea: 768,
                // タイル方式（計算部品の説明を参照）: 描画部品を常に tile の大きさで動かし、大きい画像は
                //   ①tile の大きさで構図 → ②拡大して後半の工程をタイルでやり直す（hires）。メモリは 256×384 相当で済む。
                //   ただし PC で確かめた画質が不十分なため既定では使わない（diag-gen.html?tile=1 で試せる）:
                //     全身 … 構図を 256×384 で作ると人物が幼い体つきになりやすい / 上半身 … 顔が大写しになり、タイルごとに別の顔を描き込んで崩れる
                //   hires.strength: 0.55 だと描き込みが強すぎて体や服が別の物に化けることがあった → 0.45
                tile: { w: 256, h: 384 }, hires: { strength: 0.45, steps: 20 }, tileOn: false,
                // 顔の描き直し: 顔を検出し、その周りを全体サイズまで拡大して後半の工程だけやり直す（generate の opts.face で有効）
                //   strength: やり直す割合 / steps: 描き直し用の工程数（実際に回るのは steps×strength）
                //   expand: 顔の枠を何倍に広げて切り出すか / maxZoom・minZoom: 拡大率の上限と、これ未満なら描き直さない下限
                //   実機で、顔が大きめの画像を広い範囲で描き直して体とつながらなくなった疑いがあったため、
                //   顔が小さい時（2倍以上に拡大できる時）だけ描き直し、結果の顔の位置・大きさも確かめる（計算部品側）
                // 下準備: 本番の描画の前に、この大きさで描画部品を1回動かしてシェーダーを先に作る（開いた直後の強制終了対策）
                warmup: { w: 128, h: 192 },
                face: { strength: 0.5, steps: 16, expand: 2.2, maxZoom: 4, minZoom: 2 },
                faceTiled: { strength: 0.5, steps: 16, expand: 1.8, maxZoom: 4, minZoom: 1.4 },   // タイル方式の時（顔は tile の大きさ1枚で描く）
                faceModel: 'models/yunet/face_detection_yunet_2023mar.onnx',
                preferLight: /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1),
                useWorker: true,      // Worker で実行（重みを1個ずつ読む・終わったら Worker ごと捨てる）。不可なら従来どおり
                stream: true };

  const loadScript = src => new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.async = true; s.onload = res; s.onerror = () => rej(new Error('読み込み失敗: ' + src));
    document.head.appendChild(s);
  });

  // ── 端末の対応状況 ─────────────────────────────────────────
  let _support = null;
  function supported() {
    if (_support) return _support;
    _support = (async () => {
      if (!navigator.gpu) return { ok: false, reason: 'WebGPU に対応していません' };
      if (!(navigator.storage && navigator.storage.getDirectory)) return { ok: false, reason: '端末への保存(OPFS)に対応していません' };
      try {
        const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!ad) return { ok: false, reason: 'GPU を取得できません' };
        if (cfg.ep === 'webgpu' && !ad.features.has('shader-f16') && Object.values(cfg.tiers).some(t => t.dtype === 'float16'))
          return { ok: false, reason: '半精度(f16)計算に対応していません' };
        return { ok: true };
      } catch (e) { return { ok: false, reason: String(e.message || e) }; }
    })();
    return _support;
  }

  // ── 保存状況 ─────────────────────────────────────────────
  async function dirOf(sub, create) {
    let d = await navigator.storage.getDirectory();
    for (const p of [ROOT_DIR, ...sub.split('/')].filter(Boolean)) d = await d.getDirectoryHandle(p, { create: !!create });
    return d;
  }
  async function hasFile(dir, f) {
    try {
      const mark = await (await dir.getFileHandle(f.name + '.ok')).getFile();
      return (await mark.text()) === f.sha256;
    } catch (e) { return false; }
  }
  async function tierReady(id) {
    try {
      const d = await dirOf(id), c = await dirOf('common');
      for (const f of cfg.tiers[id].files) if (!(await hasFile(d, f))) return false;
      for (const f of cfg.common) if (!(await hasFile(c, f))) return false;
      return true;
    } catch (e) { return false; }
  }
  let _ready = { M: false, S: false };
  async function status() {
    const tiers = {};
    for (const id of Object.keys(cfg.tiers)) {
      const ready = await tierReady(id);
      _ready[id] = ready;
      tiers[id] = { ready, label: cfg.tiers[id].label, short: cfg.tiers[id].short,
                    bytes: cfg.tiers[id].files.reduce((a, f) => a + f.size, 0) + cfg.common.reduce((a, f) => a + f.size, 0) };
    }
    let usage = 0, quota = 0, persisted = false;
    try { const e = await navigator.storage.estimate(); usage = e.usage || 0; quota = e.quota || 0; } catch (e) {}
    try { persisted = await navigator.storage.persisted(); } catch (e) {}
    return { tiers, readyTier: readyTier(), usage, quota, persisted };
  }
  // 使えるモデル。ふつうは標準を優先。iPhone / iPad は軽量を優先する
  //   （ページのメモリ上限が 2048MB 以下で、実機で通しの生成に成功しているのは軽量だけ。標準は描画の1工程目で強制終了した）
  const readyTier = () => cfg.preferLight ? (_ready.S ? 'S' : _ready.M ? 'M' : null) : (_ready.M ? 'M' : _ready.S ? 'S' : null);

  // ── ダウンロード（Worker） ──────────────────────────────────
  let _worker = null, _job = 0, _active = null;
  function worker() {
    if (!_worker) _worker = new Worker(cfg.assetBase + 'local-gen-worker.js' + VQ);
    return _worker;
  }
  function runJob(dir, files, onProgress) {
    return new Promise((resolve, reject) => {
      const id = ++_job, w = worker();
      const onMsg = e => {
        const m = e.data; if (m.id !== id) return;
        if (m.type === 'progress') { if (onProgress) onProgress(m); return; }
        w.removeEventListener('message', onMsg);
        if (_active && _active.id === id) _active = null;
        if (m.type === 'done') resolve();
        else { const err = new Error(m.message); if (m.aborted) err.name = 'AbortError'; reject(err); }
      };
      w.addEventListener('message', onMsg);
      _active = { id };
      w.postMessage({ cmd: 'download', id, dir: ROOT_DIR + '/' + dir,
                      files: files.map(f => ({ name: f.name, url: cfg.base + f.src, size: f.size, sha256: f.sha256 })) });
    });
  }
  async function download(id, onProgress) {
    const t = cfg.tiers[id]; if (!t) throw new Error('不明なモデル: ' + id);
    try { if (navigator.storage.persist) await navigator.storage.persist(); } catch (e) {}   // 自動削除されにくくする
    const commonTotal = cfg.common.reduce((a, f) => a + f.size, 0);
    const total = commonTotal + t.files.reduce((a, f) => a + f.size, 0);
    const fwd = off => p => onProgress && onProgress({ done: off + p.done, total, file: p.file, fileDone: p.fileDone, fileTotal: p.fileTotal });
    await runJob('common', cfg.common, fwd(0));
    await runJob(id, t.files, fwd(commonTotal));
    _ready[id] = await tierReady(id);
    if (!_ready[id]) throw new Error('保存の確認に失敗しました');
  }
  function abortDownload() { if (_active && _worker) _worker.postMessage({ cmd: 'abort', id: _active.id }); }
  async function remove(id) {
    try { const root = await dirOf(''); await root.removeEntry(id, { recursive: true }); } catch (e) {}
    _ready[id] = false;
    if (!readyTier()) { try { const root = await dirOf(''); await root.removeEntry('common', { recursive: true }); } catch (e) {} }
  }

  // ── 推論 ───────────────────────────────────────────────
  let _ortReady = null;
  function loadOrt() {
    if (!_ortReady) _ortReady = (async () => {
      if (typeof LocalGenCore === 'undefined') await loadScript(cfg.assetBase + 'local-gen-core.js' + VQ);
      if (typeof ort === 'undefined') await loadScript(cfg.assetBase + 'ort/ort.webgpu.min.js' + VQ);
      { const b = new URL(cfg.assetBase + 'ort/', location.href).href;   // ?v= は実行部品を修正した時に古いキャッシュを使わせないため
      ort.env.wasm.wasmPaths = { mjs: b + 'ort-wasm-simd-threaded.asyncify.mjs' + ORT_Q, wasm: b + 'ort-wasm-simd-threaded.asyncify.wasm' }; }
      ort.env.wasm.numThreads = 1;             // SharedArrayBuffer が無い環境（GitHub Pages）でも動くように
      ort.env.logLevel = 'error';
      try { ort.env.webgpu.powerPreference = 'high-performance'; } catch (e) {}
    })().catch(e => { _ortReady = null; throw e; });
    return _ortReady;
  }
  function makeEnv(id) {
    const t = cfg.tiers[id];
    const epOf = comp => {
      const e0 = (cfg.epFor && cfg.epFor[comp]) || cfg.ep;     // 部品ごとに実行先を変えられる（例: 文章部品だけCPU）
      // WebGPU: GPU バッファの扱いは cfg.epOptions（上の説明を参照）
      return e0 === 'webgpu' && cfg.epOptions ? Object.assign({ name: 'webgpu' }, cfg.epOptions) : e0;
    };
    const ka = cfg.ep === 'webgpu' && cfg.keepAlive !== false ? LocalGenCore.keepAlive(ort, epOf('unet')) : null;   // GPU 接続を保つ
    return {
      ort, dtype: t.dtype,
      begin: ka && ka.begin, end: ka && ka.end,
      // 部品の読み出し。設計図（重みなし・assets/ai/graphs）＋ OPFS の元ファイルを外部データとして渡す。
      //   旧方式（モデル丸ごと）は onnxruntime が重みを wasm メモリへ複製し、iPhone の上限（約1.9GB）を
      //   超えてページが強制終了していた。外部データ方式なら重みは JS から GPU へ直接転送される。
      loadModel: async comp => {
        const f = await (await (await dirOf(id)).getFileHandle(comp + '.onnx')).getFile();
        if (!cfg.useGraphs) return new Uint8Array(await f.arrayBuffer());
        const g = await fetch(`${cfg.assetBase}graphs/${id}/${comp}.onnx${VQ}`);
        if (!g.ok) throw new Error('設計図ファイルを読み込めません: ' + comp);
        const model = new Uint8Array(await g.arrayBuffer());
        return { model, externalData: [{ path: comp + '.onnx', data: new Uint8Array(await f.arrayBuffer()) }] };
      },
      loadTokenizer: async () => {
        const c = await dirOf('common');
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
  function toBlob(r) {
    const cv = document.createElement('canvas'); cv.width = r.width; cv.height = r.height;
    cv.getContext('2d').putImageData(new ImageData(r.rgba, r.width, r.height), 0, 0);
    return new Promise((res, rej) => cv.toBlob(b => b ? res(b) : rej(new Error('画像の書き出しに失敗')), 'image/png'));
  }
  const isOom = e => /memory|out of|device.*lost|lost|allocation|RangeError|OOM/i.test(String(e && (e.message || e)));

  // ── Worker で実行（iPhone の強制終了対策） ───────────────────
  //   iPhone の Safari は「JS＋GPU」の合計メモリに上限があり、モデルファイルを丸ごと読むと
  //   GPU へ送った後も写しが残って上限を超えていた。Worker では重みを1個ずつ読み、
  //   1枚ごとに Worker を捨てるので、メモリが確実に返る
  const WORKER = () => cfg.assetBase + 'local-gen-run-worker.js' + VQ;
  let _probe = null;
  function workerProbe() {
    if (_probe) return _probe;
    _probe = new Promise(res => {
      let w;
      try { w = new Worker(WORKER()); } catch (e) { res({ ok: false, reason: 'Worker を作れません' }); return; }
      const end = r => { clearTimeout(t); try { w.terminate(); } catch (e) {} res(r); };
      const t = setTimeout(() => end({ ok: false, reason: 'Worker が応答しません' }), 10000);
      w.onerror = e => end({ ok: false, reason: 'Worker エラー: ' + (e.message || '') });
      w.onmessage = e => {
        const d = e.data, needF16 = Object.values(cfg.tiers).some(t => t.dtype === 'float16');
        const ok = d.sync && (cfg.ep !== 'webgpu' || (d.gpu && d.adapter && (!needF16 || d.f16)));
        end(Object.assign({ ok, reason: ok ? '' : !d.sync ? 'Worker で同期ファイル読み込み不可' : !d.gpu ? 'Worker で WebGPU 不可' : 'Worker で GPU/半精度 不可' }, d));
      };
      w.postMessage({ cmd: 'probe', ep: cfg.ep });
    });
    return _probe;
  }
  function runInWorker(id, o, size, steps, extra) {
    return new Promise((resolve, reject) => {
      const w = new Worker(WORKER());
      const fin = () => { try { w.terminate(); } catch (e) {} if (o.signal) o.signal.removeEventListener('abort', onAbort); };
      const onAbort = () => { fin(); const e = new Error('中断しました'); e.name = 'AbortError'; reject(e); };
      if (o.signal) { if (o.signal.aborted) { onAbort(); return; } o.signal.addEventListener('abort', onAbort); }
      w.onmessage = e => {
        const d = e.data;
        if (d.type === 'progress') { if (o.onProgress) o.onProgress(d.ph, d.f); }
        else if (d.type === 'log') { if (o.onLog) o.onLog(d.text); }
        else if (d.type === 'done') { fin(); resolve(d.r); }
        else if (d.type === 'error') { fin(); const er = new Error(d.message); er.name = d.name || 'Error'; reject(er); }
      };
      w.onerror = e => { fin(); reject(new Error('Worker エラー: ' + (e.message || ''))); };
      const abs = p => new URL(cfg.assetBase + p, location.href).href;
      w.postMessage(Object.assign({ cmd: 'generate', tier: id, root: ROOT_DIR, dtype: cfg.tiers[id].dtype,
        ortBase: abs('ort/'), mjsQuery: ORT_Q, coreUrl: abs('local-gen-core.js' + VQ), graphBase: abs(`graphs/${id}/`), assetQuery: VQ, expectVersion: AV,
        ep: cfg.ep, epOptions: cfg.epOptions, epFor: cfg.epFor, stream: cfg.stream, uploadMode: cfg.uploadMode, keepAlive: cfg.keepAlive, pace: cfg.pace, settle: cfg.settle, vaeMaxArea: cfg.vaeMaxArea, debug: !!o.onLog,
        faceModelUrl: o.face ? abs(cfg.faceModel) : null,
        opts: { prompt: o.prompt, negative: o.negative, seed: o.seed, steps, guidance: o.guidance, width: size.w, height: size.h,
                tile: cfg.tileOn ? cfg.tile : null, hires: cfg.hires, warmup: cfg.warmup,
                face: o.face ? Object.assign({}, cfg.tileOn ? cfg.faceTiled : cfg.face, o.face) : null } }, extra || {}));
    });
  }
  // 診断用: 部品を選んで Worker で動かす（diag-gen.html）
  async function diagRun(tier, plan, onLog) {
    const id = _ready[tier] ? tier : readyTier();
    if (!id) throw new Error('端末内生成のモデルが保存されていません');
    const ex = { cmd: 'diag', plan };
    if (plan.pace !== undefined) ex.pace = plan.pace;          // 診断では手順ごとに切り替えられる
    if (plan.settle !== undefined) ex.settle = plan.settle;
    return (await runInWorker(id, { onLog }, { w: 0, h: 0 }, 0, ex)).diag;
  }

  // 診断用: 顔の検出だけを Worker で動かす（rgba は Uint8ClampedArray。モデルが未保存でも動く）
  async function detectFace(rgba, w, h) {
    const id = readyTier() || Object.keys(cfg.tiers)[0];
    return runInWorker(id, {}, { w: 0, h: 0 }, 0, { cmd: 'face', rgba, w, h, faceModelUrl: new URL(cfg.assetBase + cfg.faceModel, location.href).href });
  }

  // opts: { prompt, negative, seed, quality, signal, onProgress(phase, frac), face: { prompt } | null }
  async function generate(o) {
    const id = (o.tier && _ready[o.tier]) ? o.tier : readyTier();
    if (!id) throw new Error('端末内生成のモデルが保存されていません');
    const plan = Object.assign({}, PLANS[o.quality] || PLANS.balanced, o.size || {});
    const steps = o.steps || plan.steps;
    const useW = cfg.useWorker && cfg.useGraphs && (await workerProbe()).ok;
    if (!useW) await loadOrt();
    const run = size => useW ? runInWorker(id, o, size, steps) : LocalGenCore.generate(makeEnv(id), {
      prompt: o.prompt, negative: o.negative, seed: o.seed, steps, guidance: o.guidance,
      width: size.w, height: size.h, tile: cfg.tileOn ? cfg.tile : null, hires: cfg.hires, warmup: cfg.warmup, signal: o.signal, onProgress: o.onProgress });
    let r;
    try { r = await run(plan); }
    catch (e) {
      if (e.name === 'AbortError' || !isOom(e) || plan.w <= FALLBACK_SIZE.w) throw e;
      console.warn('[LGO] 端末内生成: メモリ不足のため縮小して再試行', e);
      r = await run(FALLBACK_SIZE);
    }
    // detail: 描き直した顔（rect は元画像上の位置。アプリ側で拡大後の画像に貼る）
    const detail = r.detail ? { blob: await toBlob(r.detail), rect: r.detail.rect, width: r.width, height: r.height } : null;
    return { blob: await toBlob(r), width: r.width, height: r.height, tier: id, label: cfg.tiers[id].label, detail,
             msPerStep: r.msPerStep, ms: r.ms, truncated: r.truncated, via: useW ? 'worker' : 'main' };
  }

  // オフラインで初めて生成する時に備え、実行部品を先に取得しておく（Service Worker が保存する）
  async function warmup() {
    const files = ['local-gen.js' + VQ, 'local-gen-worker.js' + VQ, 'local-gen-run-worker.js' + VQ, 'local-gen-core.js' + VQ,
                   'ort/ort.webgpu.min.js' + VQ, 'ort/ort-wasm-simd-threaded.asyncify.mjs' + ORT_Q, 'ort/ort-wasm-simd-threaded.asyncify.wasm',
                   ...['M', 'S'].flatMap(t => ['text_encoder', 'unet', 'vae_decoder'].map(c => `graphs/${t}/${c}.onnx${VQ}`)),
                   'tf.min.js', 'tf-backend-webgpu.min.js', 'upscaler.min.js'];
    await Promise.all(files.map(f => fetch(cfg.assetBase + f).then(r => r.arrayBuffer()).catch(() => null)));
  }

  window.LOCAL_GEN = {
    supported, status, download, abortDownload, remove, generate, warmup, workerProbe, diagRun, detectFace, version: AV,
    isReady: () => !!readyTier(), readyTier, plans: PLANS, tiled: () => !!cfg.tileOn,
    _configure: o => { Object.assign(cfg, o); _support = null; _probe = null; _ready = {}; for (const k of Object.keys(cfg.tiers)) _ready[k] = false; },
  };
})();
