/* ════════════════════════════════════════════════════════════
   端末内（オフライン）画像生成エンジン ── コア
     Stable Diffusion 系（tiny-sd）を ONNX Runtime で動かす最小パイプライン。
       CLIP トークナイザ → テキストエンコーダ → UNet（DPM-Solver++ 2M）→ VAE デコーダ
     メモリ上限の厳しい iPhone 向けに、3つの部品を「1つずつ読み込み → 使い終わったら解放」する。
     ブラウザ（onnxruntime-web）でも Node（onnxruntime-node・検証用）でも同じコードが動く。
   ════════════════════════════════════════════════════════════ */
(function (root) {
  'use strict';

  // ── 半精度(fp16) ⇔ 単精度 ──────────────────────────────────
  const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
  function f32ToF16Bits(v) {
    _f32[0] = v;
    const x = _u32[0], sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 0xff) - 127 + 15, mant = x & 0x7fffff;
    if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // NaN/Inf
    if (exp >= 0x1f) return sign | 0x7c00;                                            // 桁あふれ→Inf
    if (exp <= 0) {                                                                   // 非正規化数
      if (exp < -10) return sign;
      mant |= 0x800000;
      const shift = 14 - exp;
      let m = mant >>> shift;
      const rem = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
      if (rem > half || (rem === half && (m & 1))) m++;
      return sign | m;
    }
    let m = mant >>> 13;
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (m & 1))) { m++; if (m === 0x400) { m = 0; exp++; if (exp >= 0x1f) return sign | 0x7c00; } }
    return sign | (exp << 10) | m;
  }
  const F16_TABLE = (() => {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
      t[h] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
    }
    return t;
  })();
  // 出力テンソルの中身を Float32Array に（ORT は fp16 を Uint16Array のビット列か Float16Array で返す）
  function toF32(data) {
    if (data instanceof Float32Array) return data;
    if (data instanceof Uint16Array) { const o = new Float32Array(data.length); for (let i = 0; i < data.length; i++) o[i] = F16_TABLE[data[i]]; return o; }
    return Float32Array.from(data);
  }
  function fromF32(ort, dtype, f32, dims) {
    if (dtype === 'float32') return new ort.Tensor('float32', f32, dims);
    const b = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) b[i] = f32ToF16Bits(f32[i]);
    return new ort.Tensor('float16', b, dims);
  }

  // ── 乱数（シード固定で同じ画像を再現できる）──────────────────
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gaussian(n, seed) {
    const rnd = mulberry32(seed >>> 0), out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u = Math.max(rnd(), 1e-12), v = rnd(), r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  }

  // ════════════════════════════════════════════════════════════
  //  CLIP トークナイザ（HF CLIPTokenizer と同じ BPE）
  // ════════════════════════════════════════════════════════════
  const BOS = 49406, EOS = 49407, MAX_LEN = 77;
  const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const m = new Array(256);
    bs.forEach((b, i) => { m[b] = String.fromCharCode(cs[i]); });
    return m;
  }
  class CLIPTokenizer {
    constructor(vocab, mergesText) {
      this.encoder = vocab;
      this.ranks = new Map();
      const lines = mergesText.split(/\r?\n/);          // 配布ファイルは CRLF のことがある
      let r = 0;
      for (const line of lines) {
        if (!line || line.startsWith('#version')) continue;
        const p = line.split(' ');
        if (p.length === 2) this.ranks.set(p[0] + ' ' + p[1], r++);
      }
      this.byteMap = bytesToUnicode();
      this.cache = new Map();
      this.enc = new TextEncoder();
    }
    bpe(token) {
      if (this.cache.has(token)) return this.cache.get(token);
      let word = Array.from(token);
      word[word.length - 1] += '</w>';
      if (word.length === 1) { this.cache.set(token, word); return word; }
      for (;;) {
        let best = null, bestRank = Infinity;
        for (let i = 0; i < word.length - 1; i++) {
          const r = this.ranks.get(word[i] + ' ' + word[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best === null) break;
        const a = word[best], b = word[best + 1], merged = [];
        for (let i = 0; i < word.length;) {
          if (i < word.length - 1 && word[i] === a && word[i + 1] === b) { merged.push(a + b); i += 2; }
          else { merged.push(word[i]); i++; }
        }
        word = merged;
        if (word.length === 1) break;
      }
      this.cache.set(token, word);
      return word;
    }
    // テキスト → BPE トークンID（BOS/EOS なし）
    tokenize(text) {
      const clean = String(text || '').normalize('NFC').replace(/\s+/g, ' ').trim().toLowerCase();
      const ids = [];
      for (const m of clean.matchAll(PAT)) {
        const t = m[0];
        if (t === '<|startoftext|>') { ids.push(BOS); continue; }
        if (t === '<|endoftext|>')   { ids.push(EOS); continue; }
        let u = '';
        for (const b of this.enc.encode(t)) u += this.byteMap[b];
        for (const p of this.bpe(u)) { const id = this.encoder[p]; if (id !== undefined) ids.push(id); }
      }
      return ids;
    }
    // 77 トークンに揃えた入力ID（超過分は末尾から切り捨て）
    encode(text) {
      const body = this.tokenize(text).slice(0, MAX_LEN - 2);
      const ids = [BOS, ...body, EOS];
      while (ids.length < MAX_LEN) ids.push(EOS);
      return { ids, used: body.length, truncated: this.tokenize(text).length > MAX_LEN - 2 };
    }
  }

  // ════════════════════════════════════════════════════════════
  //  DPM-Solver++ 2M（diffusers DPMSolverMultistepScheduler の既定設定と同じ式）
  //    beta: scaled_linear 0.00085〜0.012 / timestep_spacing: linspace / final_sigmas_type: zero
  // ════════════════════════════════════════════════════════════
  const roundHalfEven = x => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };
  function makeScheduler(steps, cfg) {
    const c = Object.assign({ train: 1000, betaStart: 0.00085, betaEnd: 0.012 }, cfg || {});
    const ac = new Float64Array(c.train);
    let prod = 1;
    const a = Math.sqrt(c.betaStart), b = Math.sqrt(c.betaEnd);
    for (let i = 0; i < c.train; i++) {
      const beta = Math.pow(a + (b - a) * i / (c.train - 1), 2);
      prod *= (1 - beta); ac[i] = prod;
    }
    const timesteps = [];
    for (let i = 0; i <= steps; i++) timesteps.push(roundHalfEven((c.train - 1) * i / steps));
    timesteps.reverse(); timesteps.pop();
    const sigmas = timesteps.map(t => Math.sqrt((1 - ac[t]) / ac[t]));
    sigmas.push(0);
    const alphaSigma = s => { const al = 1 / Math.sqrt(s * s + 1); return [al, s * al]; };
    const lambda = s => { if (s === 0) return Infinity; const [al, sg] = alphaSigma(s); return Math.log(al) - Math.log(sg); };
    const prevX0 = [];
    return {
      timesteps,
      sigmas,
      // きれいな latent に、工程 i 開始時点のノイズを載せる（途中から描き直す時に使う）
      addNoise(i, x0, noise) {
        const [al, sg] = alphaSigma(sigmas[i]), out = new Float32Array(x0.length);
        for (let k = 0; k < out.length; k++) out[k] = al * x0[k] + sg * noise[k];
        return out;
      },
      // eps（ノイズ予測）と現在の latent から次の latent を返す
      step(i, eps, sample) {
        const sS = sigmas[i], sT = sigmas[i + 1];
        const [alS, sgS] = alphaSigma(sS);
        const x0 = new Float32Array(sample.length);
        for (let k = 0; k < x0.length; k++) x0[k] = (sample[k] - sgS * eps[k]) / alS;
        const out = new Float32Array(sample.length);
        const firstOrder = !prevX0.length || i === timesteps.length - 1;   // 履歴が無い最初の1回（途中から始める時も）と最後は1次
        if (sT === 0) { out.set(x0); }
        else {
          const [alT, sgT] = alphaSigma(sT);
          const h = lambda(sT) - lambda(sS), e = Math.exp(-h) - 1;
          if (firstOrder) {
            for (let k = 0; k < out.length; k++) out[k] = (sgT / sgS) * sample[k] - alT * e * x0[k];
          } else {
            const h0 = lambda(sS) - lambda(sigmas[i - 1]), r0 = h0 / h, m1 = prevX0[0];
            for (let k = 0; k < out.length; k++) {
              const D1 = (x0[k] - m1[k]) / r0;
              out[k] = (sgT / sgS) * sample[k] - alT * e * x0[k] - 0.5 * alT * e * D1;
            }
          }
        }
        prevX0[0] = x0;
        return out;
      },
    };
  }

  // ════════════════════════════════════════════════════════════
  //  生成パイプライン
  //    env.ort          : onnxruntime（web / node）
  //    env.loadModel(c) : 部品名 → セッション作成に渡せるもの（Uint8Array / ファイルパス /
  //                       { model, externalData, done? }。done は作成後に呼ばれる）
  //    env.loadTokenizer(): { vocab, merges }
  //    env.sessionOptions(c, dims): セッションオプション
  //    env.dtype        : 'float16' | 'float32'
  // ════════════════════════════════════════════════════════════
  const VAE_SCALE = 0.18215;
  function abortErr() { const e = new Error('中断しました'); e.name = 'AbortError'; return e; }
  // loadModel は「モデル本体」か「設計図＋外部データ」({ model, externalData }) を返す。
  // 外部データ方式なら onnxruntime-web は重みを wasm に複製せず GPU へ直接送る（メモリ節約）。
  async function withSession(env, comp, dims, fn, prog) {
    if (prog) prog(comp + ':load', 0);
    let src = await env.loadModel(comp);
    let opts = env.sessionOptions(comp, dims), done = null;
    if (src && src.model) { opts = Object.assign({}, opts, { externalData: src.externalData }); done = src.done; src = src.model; }
    let s;
    try { s = await env.ort.InferenceSession.create(src, opts); }
    finally { if (done) { try { done(); } catch (e) {} } }   // 読み出し係の後始末（ファイルを閉じる等）
    src = null; opts = null;                       // 重みのコピーを手放して GC に回収させる
    if (env.settle) await env.settle(comp);        // 読み込みで出た一時メモリが返るのを待つ（iPhone の上限対策）
    if (prog) prog(comp + ':run', 0);              // 作成完了 → ここから実行（落ちた段階の切り分け用）
    try { return await fn(s); }
    finally { try { await s.release(); } catch (e) {} s = null; }
  }

  // ── 仕上げ（VAE）のタイル分割 ──────────────────────────────
  //   VAE は1回の実行で大量の GPU メモリを使う（軽量モデルの計測: 256×384 で作成合計 621MB、448×640 なら約1.8GB）。
  //   iPhone はページのメモリ上限が 2048MB なので、大きい画像は潜在表現をタイルに分けて順に復元し、
  //   重なり部分をなだらかに混ぜてつなぐ。maxArea は1タイルの潜在表現の画素数の上限（0 や未指定なら分割しない）
  function planTiles(lw, lh, maxArea, ov) {
    if (!maxArea || lw * lh <= maxArea) return null;
    let best = null;
    for (let nx = 1; nx <= 6; nx++) for (let ny = 1; ny <= 6; ny++) {
      const tw = Math.min(lw, Math.ceil((lw + (nx - 1) * ov) / nx)), th = Math.min(lh, Math.ceil((lh + (ny - 1) * ov) / ny));
      if (tw * th > maxArea) continue;
      if (!best || nx * ny < best.nx * best.ny || (nx * ny === best.nx * best.ny && tw * th < best.tw * best.th)) best = { nx, ny, tw, th };
    }
    if (!best) return null;
    const pos = (n, full, t) => Array.from({ length: n }, (_, i) => n === 1 ? 0 : Math.round(i * (full - t) / (n - 1)));
    best.xs = pos(best.nx, lw, best.tw); best.ys = pos(best.ny, lh, best.th);
    return best;
  }
  // 1方向の重み: 隣のタイルと重なる側だけ 0→1 になだらかに変える（画像の端は 1 のまま）
  function ramp(len, start, full, lapBefore, lapAfter) {
    const w = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      let v = 1;
      if (start > 0 && i < lapBefore) v = Math.min(v, (i + 0.5) / lapBefore);
      if (start + len < full && i >= len - lapAfter) v = Math.min(v, (len - i - 0.5) / lapAfter);
      w[i] = v;
    }
    return w;
  }
  async function decodeTiled(env, ort, dt, lat, lw, lh, plan, prog) {
    const W = lw * 8, H = lh * 8, P = W * H, { tw, th, xs, ys } = plan, TW = tw * 8, TH = th * 8, TP = TW * TH;
    const acc = new Float32Array(3 * P), wsum = new Float32Array(P), total = xs.length * ys.length;
    await withSession(env, 'vae_decoder', { batch: 1, h: th, w: tw }, async s => {
      let done = 0;
      for (let iy = 0; iy < ys.length; iy++) for (let ix = 0; ix < xs.length; ix++) {
        const x0 = xs[ix], y0 = ys[iy], tile = new Float32Array(4 * th * tw);
        for (let c = 0; c < 4; c++) for (let y = 0; y < th; y++)
          tile.set(lat.subarray(c * lw * lh + (y0 + y) * lw + x0, c * lw * lh + (y0 + y) * lw + x0 + tw), c * th * tw + y * tw);
        const out = toF32((await s.run({ latent_sample: fromF32(ort, dt, tile, [1, 4, th, tw]) })).sample.data);
        const lapL = ix > 0 ? (xs[ix - 1] + tw - x0) * 8 : 0, lapR = ix < xs.length - 1 ? (x0 + tw - xs[ix + 1]) * 8 : 0;
        const lapT = iy > 0 ? (ys[iy - 1] + th - y0) * 8 : 0, lapB = iy < ys.length - 1 ? (y0 + th - ys[iy + 1]) * 8 : 0;
        const wx = ramp(TW, x0 * 8, W, lapL, lapR), wy = ramp(TH, y0 * 8, H, lapT, lapB);
        for (let y = 0; y < TH; y++) {
          const row = (y0 * 8 + y) * W + x0 * 8;
          for (let x = 0; x < TW; x++) {
            const w = wx[x] * wy[y], q = row + x, t = y * TW + x;
            acc[q] += out[t] * w; acc[P + q] += out[TP + t] * w; acc[2 * P + q] += out[2 * TP + t] * w; wsum[q] += w;
          }
        }
        if (env.settle) await env.settle('tile');   // タイルごとに使い終わったメモリを返す
        prog('vae', ++done / (total + 1));
      }
    }, prog);
    for (let q = 0; q < P; q++) { const w = wsum[q] || 1; acc[q] /= w; acc[P + q] /= w; acc[2 * P + q] /= w; }
    return acc;
  }

  // ── 顔の検出（YuNet・232KB・CPU/wasm で実行）─────────────────
  //   入力 640×640（BGR・0〜255）。出力は刻み 8/16/32 ごとの 確度(cls×obj)・枠。最も確かな顔を1つ返す
  async function detectFace(env, rgba, W, H) {
    const S = 640, sc = Math.min(S / W, S / H), nw = Math.round(W * sc), nh = Math.round(H * sc);
    const inp = new Float32Array(3 * S * S);
    for (let y = 0; y < nh; y++) {
      const fy = Math.min(H - 1, (y + 0.5) / sc - 0.5), y0 = Math.max(0, Math.floor(fy)), y1 = Math.min(H - 1, y0 + 1), ty = Math.max(0, fy - y0);
      for (let x = 0; x < nw; x++) {
        const fx = Math.min(W - 1, (x + 0.5) / sc - 0.5), x0 = Math.max(0, Math.floor(fx)), x1 = Math.min(W - 1, x0 + 1), tx = Math.max(0, fx - x0);
        for (let c = 0; c < 3; c++) {
          const a = rgba[(y0 * W + x0) * 4 + c], b = rgba[(y0 * W + x1) * 4 + c], d = rgba[(y1 * W + x0) * 4 + c], e = rgba[(y1 * W + x1) * 4 + c];
          inp[(2 - c) * S * S + y * S + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (d * (1 - tx) + e * tx) * ty;   // RGB → BGR
        }
      }
    }
    const s = await env.ort.InferenceSession.create(await env.loadFaceModel(), env.faceSessionOptions ? env.faceSessionOptions() : {});
    let out;
    try { out = await s.run({ [s.inputNames[0]]: new env.ort.Tensor('float32', inp, [1, 3, S, S]) }); }
    finally { try { await s.release(); } catch (e) {} }
    let best = null;
    for (const st of [8, 16, 32]) {
      const cls = out['cls_' + st].data, obj = out['obj_' + st].data, bb = out['bbox_' + st].data, cols = S / st, rows = S / st;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const i = r * cols + c, score = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
        if (score < 0.6 || (best && score <= best.score)) continue;
        const cx = (c + bb[i * 4]) * st, cy = (r + bb[i * 4 + 1]) * st, w = Math.exp(bb[i * 4 + 2]) * st, h = Math.exp(bb[i * 4 + 3]) * st;
        best = { x: (cx - w / 2) / sc, y: (cy - h / 2) / sc, w: w / sc, h: h / sc, score };
      }
    }
    return best;
  }
  // latent（sw×sh）の一部（小数座標の矩形 x0,y0,cw,ch）を dw×dh に引き伸ばす（双一次補間）
  function resampleLatent(lat, sw, sh, x0, y0, cw, ch, dw, dh) {
    if (dw == null) { dw = sw; dh = sh; }
    const out = new Float32Array(4 * dw * dh), P = dw * dh, SP = sw * sh;
    for (let y = 0; y < dh; y++) {
      const fy = Math.min(sh - 1, Math.max(0, y0 + (y + 0.5) * ch / dh - 0.5)), ya = Math.floor(fy), yb = Math.min(sh - 1, ya + 1), ty = fy - ya;
      for (let x = 0; x < dw; x++) {
        const fx = Math.min(sw - 1, Math.max(0, x0 + (x + 0.5) * cw / dw - 0.5)), xa = Math.floor(fx), xb = Math.min(sw - 1, xa + 1), tx = fx - xa;
        for (let c = 0; c < 4; c++) {
          const o = c * SP;
          out[c * P + y * dw + x] = (lat[o + ya * sw + xa] * (1 - tx) + lat[o + ya * sw + xb] * tx) * (1 - ty)
                                  + (lat[o + yb * sw + xa] * (1 - tx) + lat[o + yb * sw + xb] * tx) * ty;
        }
      }
    }
    return out;
  }
  // 1方向のタイルの開始位置: 長さ full を、大きさ t のタイルで、重なり minOv 以上になるよう等間隔に覆う
  function tilePos(full, t, minOv) {
    if (full <= t) return [0];
    const nt = Math.ceil((full - minOv) / (t - minOv));
    return Array.from({ length: nt }, (_, i) => Math.round(i * (full - t) / (nt - 1)));
  }

  // env.begin / env.end があれば生成の前後に呼ぶ（GPU 接続を最後まで保つ「つなぎ」用。下記）
  async function generate(env, o) {
    if (env.begin) await env.begin();
    try { return await generateInner(env, o); }
    finally { if (env.end) { try { await env.end(); } catch (e) {} } }
  }
  async function generateInner(env, o) {
    const W = o.width || 448, H = o.height || 640;
    if (W % 32 || H % 32) throw new Error('幅・高さは32の倍数にしてください');
    const steps = Math.max(2, o.steps || 18), g = o.guidance == null ? 7 : o.guidance;
    const lw = W / 8, lh = H / 8, n = 4 * lw * lh;
    const ort = env.ort, dt = env.dtype || 'float16';
    const prog = (phase, f) => { if (o.onProgress) o.onProgress(phase, f); };
    const check = () => { if (o.signal && o.signal.aborted) throw abortErr(); };
    const t0 = Date.now();

    // ① 文字 → トークン
    if (!env._tok) { const t = await env.loadTokenizer(); env._tok = new CLIPTokenizer(t.vocab, t.merges); }
    const neg = env._tok.encode(o.negative || ''), pos = env._tok.encode(o.prompt || '');
    const face = o.face && env.loadFaceModel ? o.face : null;          // 顔の描き直し（下記 refineFace）
    const fpos = face ? env._tok.encode(face.prompt || o.prompt || '') : null;
    check();

    // ② テキストエンコーダ（無条件・条件をまとめて1回で）
    prog('text', 0);
    // 文章部品は常に batch 2（第15回まで実機で成功していた形）。顔用の指示文は同じセッションでもう1回実行する
    //   （16a〜16c は batch 3 にしたあと、448×640 の描画1工程目で3回続けて強制終了したため戻す）
    const ids = new BigInt64Array(2 * MAX_LEN);
    neg.ids.forEach((v, i) => { ids[i] = BigInt(v); });
    pos.ids.forEach((v, i) => { ids[MAX_LEN + i] = BigInt(v); });
    const hs = MAX_LEN * 768;
    let fhid = null;
    const hid = await withSession(env, 'text_encoder', { batch: 2 }, async s => {
      const h = toF32((await s.run({ input_ids: new ort.Tensor('int64', ids, [2, MAX_LEN]) })).last_hidden_state.data);
      if (face) {
        const fids = new BigInt64Array(2 * MAX_LEN);
        fpos.ids.forEach((v, i) => { fids[i] = BigInt(v); fids[MAX_LEN + i] = BigInt(v); });
        fhid = toF32((await s.run({ input_ids: new ort.Tensor('int64', fids, [2, MAX_LEN]) })).last_hidden_state.data).slice(0, hs);
      }
      return h;
    }, prog);
    const ctxUncond = fromF32(ort, dt, hid.subarray(0, hs), [1, MAX_LEN, 768]);
    const ctxCond   = fromF32(ort, dt, hid.subarray(hs, 2 * hs), [1, MAX_LEN, 768]);
    const ctxFace   = face ? fromF32(ort, dt, fhid, [1, MAX_LEN, 768]) : null;
    fhid = null;
    prog('text', 1); check();

    // ③ UNet で少しずつノイズを取り除く（CFG は1枚ずつ2回推論＝メモリ節約）
    //   o.tile（{w,h} px）があり画像がそれより大きい時は「タイル方式」:
    //     描画部品は常にタイルの大きさ（例 256×384）で動かす。iPhone はページのメモリ上限 2048MB に対し、
    //     448×640 を1枚で描くと上限ぎりぎり（実機: 成功4・失敗2、すべて描画の1工程目）で、256×384 は一度も落ちていない。
    //     ① タイルの大きさで全体の構図を描く → ② 目標の大きさへ拡大し、ノイズを載せて後半の工程をやり直す。
    //     ②は latent を重なりのあるタイルに分けて1枚ずつ推論し、ノイズ予測を重なり部分でなだらかに混ぜる
    const seed = o.seed == null ? (Date.now() & 0x7fffffff) : o.seed;
    const T = o.tile && (lw > o.tile.w / 8 || lh > o.tile.h / 8) ? { w: Math.min(lw, o.tile.w / 8), h: Math.min(lh, o.tile.h / 8) } : null;
    const uw = T ? T.w : lw, uh = T ? T.h : lh, un = 4 * uw * uh;          // 描画部品（セッション）の大きさ
    // 部品を開いて最初の2回の推論のあとは、少し待ってメモリを返させる（warm）。
    //   実機の強制終了は3回とも「描画の1工程目」＝中間バッファを初めて作る時で、読み込み直後の一時メモリと重なる
    let warm = 0;
    const unet = (fn, pg) => { warm = 2; return withSession(env, 'unet', { batch: 1, h: uh, w: uw }, fn, pg); };
    const rest = async () => { if (warm > 0 && env.settle) { warm--; await env.settle('tile'); } else if (env.tick) await env.tick(); };
    // 大きさ xw×xh の latent 全体のノイズ予測（CFG 済み）。xw×xh が部品より大きければタイルに分ける
    const epsOf = async (s, x, xw, xh, ts, ctx, onTile) => {
      const one = async sampleArr => {
        const sample = fromF32(ort, dt, sampleArr, [1, 4, uh, uw]);
        const eu = toF32((await s.run({ sample, timestep: ts, encoder_hidden_states: ctxUncond })).out_sample.data);
        await rest();                                // 実行ごとに不要メモリの回収を促す
        const ec = toF32((await s.run({ sample, timestep: ts, encoder_hidden_states: ctx })).out_sample.data);
        await rest();
        const e = new Float32Array(un);
        for (let k = 0; k < un; k++) e[k] = eu[k] + g * (ec[k] - eu[k]);
        return e;
      };
      if (xw === uw && xh === uh) { const e = await one(x); if (onTile) onTile(1); return e; }
      const xs = tilePos(xw, uw, 8), ys = tilePos(xh, uh, 8), P = xw * xh, TP = uw * uh, total = xs.length * ys.length;
      const acc = new Float32Array(4 * P), wsum = new Float32Array(P);
      let done = 0;
      for (let iy = 0; iy < ys.length; iy++) for (let ix = 0; ix < xs.length; ix++) {
        check();
        const x0 = xs[ix], y0 = ys[iy], tile = new Float32Array(un);
        for (let c = 0; c < 4; c++) for (let y = 0; y < uh; y++)
          tile.set(x.subarray(c * P + (y0 + y) * xw + x0, c * P + (y0 + y) * xw + x0 + uw), c * TP + y * uw);
        const e = await one(tile);
        const lapL = ix > 0 ? xs[ix - 1] + uw - x0 : 0, lapR = ix < xs.length - 1 ? x0 + uw - xs[ix + 1] : 0;
        const lapT = iy > 0 ? ys[iy - 1] + uh - y0 : 0, lapB = iy < ys.length - 1 ? y0 + uh - ys[iy + 1] : 0;
        const wx = ramp(uw, x0, xw, lapL, lapR), wy = ramp(uh, y0, xh, lapT, lapB);
        for (let y = 0; y < uh; y++) for (let xx = 0; xx < uw; xx++) {
          const w = wx[xx] * wy[y], q = (y0 + y) * xw + x0 + xx, t = y * uw + xx;
          for (let c = 0; c < 4; c++) acc[c * P + q] += e[c * TP + t] * w;
          wsum[q] += w;
        }
        if (onTile) onTile(++done / total);
      }
      for (let q = 0; q < P; q++) { const w = wsum[q] || 1; for (let c = 0; c < 4; c++) acc[c * P + q] /= w; }
      return acc;
    };
    // 工程 from から最後まで（s は開いている描画部品）。onStep(0〜1)
    const denoise = async (s, x, xw, xh, sc, from, ctx, onStep) => {
      const N = sc.timesteps.length - from;
      for (let i = from; i < sc.timesteps.length; i++) {
        check();
        const ts = fromF32(ort, dt, new Float32Array([sc.timesteps[i]]), []);
        const eps = await epsOf(s, x, xw, xh, ts, ctx, f => onStep((i - from + f) / N));
        x = sc.step(i, eps, x);
      }
      return x;
    };
    const tilesOf = (xw, xh) => tilePos(xw, uw, 8).length * tilePos(xh, uh, 8).length;
    const tUnet = Date.now();
    let lat = await unet(async s => {
      if (!T) return denoise(s, gaussian(n, seed), lw, lh, makeScheduler(steps), 0, ctxCond, f => prog('unet', f));
      // ① 構図（タイルの大きさ）→ ② 拡大して後半をタイルでやり直す。進み具合は推論回数の割合で出す
      const hi = Object.assign({ strength: 0.55, steps }, o.hires || {});
      const hsteps = Math.max(4, hi.steps), hfrom = Math.min(hsteps - 2, Math.max(0, Math.round(hsteps * (1 - hi.strength))));
      const w1 = steps, w2 = (hsteps - hfrom) * tilesOf(lw, lh), share = w1 / (w1 + w2);
      const base = await denoise(s, gaussian(un, seed), uw, uh, makeScheduler(steps), 0, ctxCond, f => prog('unet', f * share));
      const hsch = makeScheduler(hsteps);
      const start = hsch.addNoise(hfrom, resampleLatent(base, uw, uh, 0, 0, uw, uh, lw, lh), gaussian(n, (seed ^ 0x2545f491) >>> 0));
      if (env.log) env.log(`タイル方式: 構図 ${uw * 8}×${uh * 8}・${steps}工程 → ${W}×${H} を ${tilesOf(lw, lh)}枚のタイルで ${hsteps - hfrom}工程`);
      return denoise(s, start, lw, lh, hsch, hfrom, ctxCond, f => prog('unet', share + (1 - share) * f));
    }, prog);
    const msUnet = Date.now() - tUnet;
    check();

    // ④ VAE デコーダで画像へ
    prog('vae', 0);
    const decode = async (x0, xw, xh, pg) => {       // latent（xw×xh）→ RGBA（xw*8 × xh*8）
      const m = 4 * xw * xh, plane = xw * xh * 64, z = new Float32Array(m);
      for (let k = 0; k < m; k++) z[k] = x0[k] / VAE_SCALE;
      const tiles = planTiles(xw, xh, env.vaeMaxArea, 8);
      const img = tiles ? await decodeTiled(env, ort, dt, z, xw, xh, tiles, pg)
        : await withSession(env, 'vae_decoder', { batch: 1, h: xh, w: xw }, async s => {
          const out = await s.run({ latent_sample: fromF32(ort, dt, z, [1, 4, xh, xw]) });
          return toF32(out.sample.data);
        }, pg);
      const px = new Uint8ClampedArray(plane * 4);
      for (let p = 0; p < plane; p++) {
        px[p * 4]     = (img[p] / 2 + 0.5) * 255 + 0.5;
        px[p * 4 + 1] = (img[plane + p] / 2 + 0.5) * 255 + 0.5;
        px[p * 4 + 2] = (img[2 * plane + p] / 2 + 0.5) * 255 + 0.5;
        px[p * 4 + 3] = 255;
      }
      return px;
    };
    const rgba = await decode(lat, lw, lh, prog);
    prog('vae', 1);

    // ⑤ 顔の描き直し（全身の構図では顔が小さく、そのままでは崩れやすい）
    //    顔を検出 → その周りの latent を描画部品の大きさ（uw×uh）に引き伸ばす → ノイズを載せて後半の工程だけやり直す → 画像にして返す。
    //    顔は必ず1枚で描く（タイルに分けると、タイルごとに違う顔を描いて二重にぶれる。PC で確認）。失敗しても全体の画像はそのまま返す
    let detail = null;
    if (face) {
      try { detail = await refineFace(); }
      catch (e) { if (e && e.name === 'AbortError') throw e; if (env.log) env.log('顔の描き直しを省略: ' + String(e && e.message || e).slice(0, 200)); }
    }
    async function refineFace() {
      prog('face', 0);
      const box = face.box || await detectFace(env, rgba, W, H);   // face.box はテスト用（検出を飛ばして枠を指定）
      if (!box) { if (env.log) env.log('顔が見つからないため描き直しなし'); return null; }
      // 顔の枠を頭全体（髪・あご）まで広げ、描き直す画像（FW×FH）と同じ縦横比の切り出し枠にする
      const FW = uw * 8, FH = uh * 8, ex = face.expand || 1.8, cwMin = FW / (face.maxZoom || 4);
      let cw = Math.max(box.w * ex, box.h * ex * FW / FH, cwMin);
      if (cw > Math.min(W, FW / (face.minZoom || 1.4))) { if (env.log) env.log(`顔が十分大きいため描き直しなし（顔の幅 ${Math.round(box.w)}px）`); return null; }
      const chh = Math.min(H, cw * FH / FW); cw = chh * FW / FH;
      let x0 = box.x + box.w / 2 - cw / 2, y0 = box.y + box.h * 0.45 - chh / 2;
      x0 = Math.min(Math.max(0, x0), W - cw); y0 = Math.min(Math.max(0, y0), H - chh);
      const z = FW / cw;
      if (env.log) env.log(`顔を検出（幅 ${Math.round(box.w)}px・確度 ${(box.score || 1).toFixed(2)}）→ ${z.toFixed(1)}倍で描き直し`);
      const up = resampleLatent(lat, lw, lh, x0 / 8, y0 / 8, cw / 8, chh / 8, uw, uh);
      const fsteps = Math.max(4, face.steps || steps), from = Math.min(fsteps - 2, Math.max(0, Math.round(fsteps * (1 - (face.strength || 0.5)))));
      const fsch = makeScheduler(fsteps);
      const noisy = fsch.addNoise(from, up, gaussian(un, (seed ^ 0x5bd1e995) >>> 0));
      const fprog = (ph, f) => prog('face:' + ph, f);
      const flat = await unet(s => denoise(s, noisy, uw, uh, fsch, from, ctxFace, f => prog('face', 0.05 + 0.75 * f)), fprog);
      const frgba = await decode(flat, uw, uh, (ph, f) => (ph === 'vae' ? prog('face', 0.8 + 0.2 * f) : fprog(ph, f)));
      // 描き直した結果を確かめる: 顔が元と同じ位置・大きさにあること。ずれていたら体とつながらないので使わない
      if (!face.box) {
        const want = { cx: (box.x + box.w / 2 - x0) * z, cy: (box.y + box.h / 2 - y0) * z, w: box.w * z };
        const got = await detectFace(env, frgba, FW, FH);
        const okPos = got && Math.abs(got.x + got.w / 2 - want.cx) < FW * 0.2 && Math.abs(got.y + got.h / 2 - want.cy) < FH * 0.2;
        const ratio = got ? got.w / want.w : 0;
        if (!okPos || ratio < 0.6 || ratio > 1.6) {
          if (env.log) env.log(got ? `描き直した顔の位置・大きさが元と合わないため使わない（大きさ ${ratio.toFixed(2)}倍）` : '描き直した画像に顔が見つからないため使わない');
          prog('face', 1);
          return null;
        }
        if (env.log) env.log(`描き直しを確認（大きさ ${ratio.toFixed(2)}倍）`);
      }
      prog('face', 1);
      // rgba は FW×FH。rect は全体の画像（W×H）上の位置
      return { rgba: frgba, width: FW, height: FH, rect: { x: x0, y: y0, w: cw, h: chh }, box };
    }
    return { rgba, width: W, height: H, steps, ms: Date.now() - t0, msPerStep: Math.round(msUnet / steps),
             truncated: pos.truncated, tokensUsed: pos.used, detail };
  }

  // ── GPU 接続の「つなぎ」 ─────────────────────────────────
  //   onnxruntime-web 1.31 は、最後のセッションを解放すると GPU 接続（GPUDevice）を破棄し、
  //   次のセッションで作り直す（Chromium で確認）。iPhone の Safari では、文章部品→描画部品の
  //   切り替えで作り直した直後の描画が毎回強制終了していた（実機テスト第4〜6回、例外なし）。
  //   そこで生成の間は、極小のモデル（x+x）のセッションを1つ持ち続けて接続を保つ
  const KEEPALIVE_ONNX = 'CAgSBGxnbzM6PQoOCgF4CgF4EgF5IgNBZGQSCWtlZXBhbGl2ZVoPCgF4EgoKCAgBEgQKAggEYg8KAXkSCgoICAESBAoCCARCBAoAEBE=';
  function keepAlive(ortRef, ep, extraOpts) {
    let sess = null;
    return {
      begin: async () => {
        const bin = typeof atob === 'function' ? atob(KEEPALIVE_ONNX) : Buffer.from(KEEPALIVE_ONNX, 'base64').toString('binary');
        const u8 = Uint8Array.from(bin, c => c.charCodeAt(0));
        sess = await ortRef.InferenceSession.create(u8, Object.assign({ executionProviders: [ep] }, extraOpts));
      },
      end: async () => { if (sess) { const s = sess; sess = null; await s.release(); } },
    };
  }

  const VERSION = '2026-11-17a';
  const api = { CLIPTokenizer, makeScheduler, generate, gaussian, f32ToF16Bits, toF32, keepAlive, MAX_LEN, VERSION,
                _withSession: withSession, _fromF32: fromF32, _planTiles: planTiles, _detectFace: detectFace, _tilePos: tilePos };   // _ 付きは診断用
  root.LocalGenCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
