/* ════════════════════════════════════════════════════════════
   端末内生成のモデルを端末に保存する Worker（OPFS）
     ・iPhone の Safari はメインスレッドからの OPFS 書き込みが壊れるため（実機診断で確認）、
       Worker の同期アクセスハンドルで書く
     ・HTTP Range で途中から再開。完了後に SHA-256 で破損・改ざんを検出
     ・ハッシュは逐次計算（数百MBを一度にメモリへ載せない）
   メッセージ:
     → { cmd:'download', id, dir, files:[{ name, url, size, sha256 }] }
     → { cmd:'abort', id }
     ← { id, type:'progress', file, fileDone, fileTotal, done, total }
     ← { id, type:'done' } / { id, type:'error', message, aborted }
   ════════════════════════════════════════════════════════════ */
'use strict';

// ── SHA-256（逐次） ──────────────────────────────────────────
const K = new Uint32Array([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
  0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
  0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
  0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
class Sha256 {
  constructor() {
    this.h = new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
    this.buf = new Uint8Array(64); this.bufLen = 0; this.len = 0; this.w = new Uint32Array(64);
  }
  _block(p, o) {
    const w = this.w, h = this.h;
    for (let i = 0; i < 16; i++) w[i] = (p[o + i * 4] << 24) | (p[o + i * 4 + 1] << 16) | (p[o + i * 4 + 2] << 8) | p[o + i * 4 + 3];
    for (let i = 16; i < 64; i++) {
      const a = w[i - 15], b = w[i - 2];
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let A = h[0], B = h[1], C = h[2], D = h[3], E = h[4], F = h[5], G = h[6], H = h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((E >>> 6) | (E << 26)) ^ ((E >>> 11) | (E << 21)) ^ ((E >>> 25) | (E << 7));
      const t1 = (H + S1 + ((E & F) ^ (~E & G)) + K[i] + w[i]) | 0;
      const S0 = ((A >>> 2) | (A << 30)) ^ ((A >>> 13) | (A << 19)) ^ ((A >>> 22) | (A << 10));
      const t2 = (S0 + ((A & B) ^ (A & C) ^ (B & C))) | 0;
      H = G; G = F; F = E; E = (D + t1) | 0; D = C; C = B; B = A; A = (t1 + t2) | 0;
    }
    h[0] += A; h[1] += B; h[2] += C; h[3] += D; h[4] += E; h[5] += F; h[6] += G; h[7] += H;
  }
  update(d) {
    let i = 0; this.len += d.length;
    if (this.bufLen) {
      const n = Math.min(64 - this.bufLen, d.length);
      this.buf.set(d.subarray(0, n), this.bufLen); this.bufLen += n; i = n;
      if (this.bufLen === 64) { this._block(this.buf, 0); this.bufLen = 0; }
    }
    for (; i + 64 <= d.length; i += 64) this._block(d, i);
    if (i < d.length) { this.buf.set(d.subarray(i), 0); this.bufLen = d.length - i; }
  }
  hex() {
    const bits = this.len * 8, pad = new Uint8Array(((this.bufLen < 56) ? 56 : 120) - this.bufLen + 8);
    pad[0] = 0x80;
    const hi = Math.floor(bits / 4294967296), lo = bits >>> 0, n = pad.length;
    pad[n - 8] = hi >>> 24; pad[n - 7] = hi >>> 16; pad[n - 6] = hi >>> 8; pad[n - 5] = hi;
    pad[n - 4] = lo >>> 24; pad[n - 3] = lo >>> 16; pad[n - 2] = lo >>> 8; pad[n - 1] = lo;
    const keep = this.len; this.update(pad); this.len = keep;
    return Array.from(this.h, x => (x >>> 0).toString(16).padStart(8, '0')).join('');
  }
}
self.Sha256 = Sha256;   // テスト用

// ── ダウンロード ────────────────────────────────────────────
const jobs = new Map();
const CHUNK = 4 * 1048576;

async function getDir(path) {
  let d = await navigator.storage.getDirectory();
  for (const p of path.split('/').filter(Boolean)) d = await d.getDirectoryHandle(p, { create: true });
  return d;
}
async function fileOk(dir, f) {
  try {
    const mark = await (await dir.getFileHandle(f.name + '.ok')).getFile();
    return (await mark.text()) === f.sha256;
  } catch (e) { return false; }
}

async function downloadOne(id, dir, f, base, total, post, signal) {
  if (await fileOk(dir, f)) { post({ type: 'progress', file: f.name, fileDone: f.size, fileTotal: f.size, done: base + f.size, total }); return; }
  const fh = await dir.getFileHandle(f.name, { create: true });
  const h = await fh.createSyncAccessHandle();
  try {
    let have = h.getSize();
    if (have > f.size) { h.truncate(0); have = 0; }
    // 既に保存済みの部分をハッシュに通してから続きを取りに行く
    let sha = new Sha256();
    const rd = new Uint8Array(CHUNK);
    for (let off = 0; off < have; off += CHUNK) {
      const n = h.read(rd.subarray(0, Math.min(CHUNK, have - off)), { at: off });
      sha.update(rd.subarray(0, n));
    }
    let pos = have;
    if (pos < f.size) {
      const res = await fetch(f.url, { headers: pos ? { Range: `bytes=${pos}-` } : {}, signal, cache: 'no-store' });
      if (pos && res.status === 200) { h.truncate(0); pos = 0; sha = new Sha256(); }   // Range 非対応→最初から
      else if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      let lastPost = 0;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        h.write(value, { at: pos }); sha.update(value); pos += value.length;
        if (pos > f.size) throw new Error('想定より大きいファイルです');
        const now = Date.now();
        if (now - lastPost > 200) { lastPost = now; post({ type: 'progress', file: f.name, fileDone: pos, fileTotal: f.size, done: base + pos, total }); }
      }
      h.flush();
    }
    if (pos !== f.size) throw new Error(`途中で途切れました（${pos}/${f.size}）`);
    const got = sha.hex();
    if (got !== f.sha256) { h.truncate(0); h.flush(); throw new Error('ファイルが壊れていました（ハッシュ不一致）。もう一度お試しください'); }
  } finally { h.close(); }
  const mk = await (await dir.getFileHandle(f.name + '.ok', { create: true })).createSyncAccessHandle();
  try { const b = new TextEncoder().encode(f.sha256); mk.truncate(0); mk.write(b, { at: 0 }); mk.flush(); } finally { mk.close(); }
  post({ type: 'progress', file: f.name, fileDone: f.size, fileTotal: f.size, done: base + f.size, total });
}

self.onmessage = async e => {
  const m = e.data, post = x => self.postMessage({ id: m.id, ...x });
  if (m.cmd === 'abort') { const j = jobs.get(m.id); if (j) j.abort(); return; }
  if (m.cmd !== 'download') return;
  const ac = new AbortController(); jobs.set(m.id, ac);
  try {
    const dir = await getDir(m.dir);
    const total = m.files.reduce((a, f) => a + f.size, 0);
    let base = 0;
    for (const f of m.files) { await downloadOne(m.id, dir, f, base, total, post, ac.signal); base += f.size; }
    post({ type: 'done' });
  } catch (err) {
    post({ type: 'error', aborted: ac.signal.aborted, message: ac.signal.aborted ? '中断しました' : String(err && err.message || err) });
  } finally { jobs.delete(m.id); }
};
