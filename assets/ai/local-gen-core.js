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
      // eps（ノイズ予測）と現在の latent から次の latent を返す
      step(i, eps, sample) {
        const sS = sigmas[i], sT = sigmas[i + 1];
        const [alS, sgS] = alphaSigma(sS);
        const x0 = new Float32Array(sample.length);
        for (let k = 0; k < x0.length; k++) x0[k] = (sample[k] - sgS * eps[k]) / alS;
        const out = new Float32Array(sample.length);
        const firstOrder = i === 0 || i === timesteps.length - 1;
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
    if (prog) prog(comp + ':run', 0);              // 作成完了 → ここから実行（落ちた段階の切り分け用）
    try { return await fn(s); }
    finally { try { await s.release(); } catch (e) {} s = null; }
  }

  async function generate(env, o) {
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
    check();

    // ② テキストエンコーダ（無条件・条件をまとめて1回で）
    prog('text', 0);
    const ids = new BigInt64Array(2 * MAX_LEN);
    neg.ids.forEach((v, i) => { ids[i] = BigInt(v); });
    pos.ids.forEach((v, i) => { ids[MAX_LEN + i] = BigInt(v); });
    const hid = await withSession(env, 'text_encoder', { batch: 2 }, async s => {
      const out = await s.run({ input_ids: new ort.Tensor('int64', ids, [2, MAX_LEN]) });
      return toF32(out.last_hidden_state.data);
    }, prog);
    const hs = MAX_LEN * 768;
    const ctxUncond = fromF32(ort, dt, hid.subarray(0, hs), [1, MAX_LEN, 768]);
    const ctxCond   = fromF32(ort, dt, hid.subarray(hs, 2 * hs), [1, MAX_LEN, 768]);
    prog('text', 1); check();

    // ③ UNet で少しずつノイズを取り除く（CFG は1枚ずつ2回推論＝メモリ節約）
    const sch = makeScheduler(steps);
    let lat = gaussian(n, o.seed == null ? (Date.now() & 0x7fffffff) : o.seed);
    const tUnet = Date.now();
    await withSession(env, 'unet', { batch: 1, h: lh, w: lw }, async s => {
      for (let i = 0; i < steps; i++) {
        check();
        const ts = fromF32(ort, dt, new Float32Array([sch.timesteps[i]]), []);
        const sample = fromF32(ort, dt, lat, [1, 4, lh, lw]);
        const eu = toF32((await s.run({ sample, timestep: ts, encoder_hidden_states: ctxUncond })).out_sample.data);
        const ec = toF32((await s.run({ sample, timestep: ts, encoder_hidden_states: ctxCond })).out_sample.data);
        const eps = new Float32Array(n);
        for (let k = 0; k < n; k++) eps[k] = eu[k] + g * (ec[k] - eu[k]);
        lat = sch.step(i, eps, lat);
        prog('unet', (i + 1) / steps);
      }
    }, prog);
    const msUnet = Date.now() - tUnet;
    check();

    // ④ VAE デコーダで画像へ
    prog('vae', 0);
    for (let k = 0; k < n; k++) lat[k] /= VAE_SCALE;
    const img = await withSession(env, 'vae_decoder', { batch: 1, h: lh, w: lw }, async s => {
      const out = await s.run({ latent_sample: fromF32(ort, dt, lat, [1, 4, lh, lw]) });
      return toF32(out.sample.data);
    }, prog);
    const rgba = new Uint8ClampedArray(W * H * 4), plane = W * H;
    for (let p = 0; p < plane; p++) {
      rgba[p * 4]     = (img[p] / 2 + 0.5) * 255 + 0.5;
      rgba[p * 4 + 1] = (img[plane + p] / 2 + 0.5) * 255 + 0.5;
      rgba[p * 4 + 2] = (img[2 * plane + p] / 2 + 0.5) * 255 + 0.5;
      rgba[p * 4 + 3] = 255;
    }
    prog('vae', 1);
    return { rgba, width: W, height: H, steps, ms: Date.now() - t0, msPerStep: Math.round(msUnet / steps),
             truncated: pos.truncated, tokensUsed: pos.used };
  }

  const api = { CLIPTokenizer, makeScheduler, generate, gaussian, f32ToF16Bits, toF32, MAX_LEN };
  root.LocalGenCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
