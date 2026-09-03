// 運気の女神 Lucky Girl Oracle ── Service Worker
// アプリシェル(HTML/アイコン)をキャッシュしてオフライン起動＆高速再表示。
// 画像生成API(Pollinations/Horde)など外部オリジンはSWを介さずネットワークへ通す。
const CACHE = 'lg-oracle-v2';

// アプリシェル: 起動に必須のものだけ。スプラッシュ(assets/splash)は
// iOS が起動時に直接読むためSWキャッシュ不要 → 初回インストールを軽くする。
const ASSETS = [
  './', './index.html', './manifest.json',
  './assets/icons/icon-192.png',
  './assets/icons/icon-512.png',
  './assets/icons/icon-512-maskable.png',
  './assets/icons/apple-touch-icon.png',
];

self.addEventListener('install', e => {
  self.skipWaiting();
  // 1件でも失敗すると addAll 全体が落ちるため個別に投入する
  e.waitUntil(
    caches.open(CACHE).then(c => Promise.all(
      ASSETS.map(u => c.add(u).catch(() => {}))
    ))
  );
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

  let url;
  try { url = new URL(req.url); } catch (_) { return; }
  if (url.origin !== location.origin) return;   // 外部(生成API等)はブラウザ任せ

  // HTML/ナビゲーション: ネットワーク優先（更新を即反映）、失敗時にキャッシュ
  if (req.mode === 'navigate' || url.pathname.endsWith('.html')) {
    e.respondWith(
      fetch(req)
        .then(res => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then(c => c.put(req, copy)).catch(() => {});
          }
          return res;
        })
        .catch(() => caches.match(req).then(r => r || caches.match('./index.html')))
    );
    return;
  }

  // その他の同一オリジン資産: stale-while-revalidate
  //   → キャッシュがあれば即返して体感を最速に、裏で静かに最新へ更新する
  e.respondWith(
    caches.match(req).then(cached => {
      const network = fetch(req).then(res => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then(c => c.put(req, copy)).catch(() => {});
        }
        return res;
      }).catch(() => cached);
      return cached || network;
    })
  );
});
