// 運気の女神 Lucky Girl Oracle ── Service Worker
// アプリシェル(HTML/アイコン)をキャッシュしオフライン起動＆高速再表示。
// 画像生成API(Pollinations/Horde)など外部はキャッシュせずネットワークへ通す。
const CACHE = 'lg-oracle-v1';
const ASSETS = [
  './', './index.html', './manifest.json',
  './icon-192.png', './icon-512.png', './icon-512-maskable.png', './apple-touch-icon.png'
];

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS).catch(() => {})));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // 外部オリジン(画像生成API等)はSWを介さずブラウザ任せ
  if (url.origin !== location.origin) return;
  // HTML/ナビゲーションはネットワーク優先（更新を反映）、失敗時キャッシュ
  if (req.mode === 'navigate' || url.pathname.endsWith('.html')) {
    e.respondWith(
      fetch(req).then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(req, copy));
        return res;
      }).catch(() => caches.match(req).then(r => r || caches.match('./index.html')))
    );
    return;
  }
  // その他の同一オリジン資産はキャッシュ優先
  e.respondWith(
    caches.match(req).then(cached => cached || fetch(req).then(res => {
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(req, copy));
      return res;
    }).catch(() => cached))
  );
});
