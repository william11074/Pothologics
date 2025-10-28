/* CRATER Service Worker */
const VERSION = 'v2';
const APP_SHELL = [
  '/',
  '/routes',
  '/detect',
  '/map',
  '/static/style.css',
  '/work_orders.csv'
];

const RUNTIME_CACHE = `runtime-${VERSION}`;
const SHELL_CACHE = `shell-${VERSION}`;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => ![SHELL_CACHE, RUNTIME_CACHE].includes(k)).map((k) => caches.delete(k)))).then(() => self.clients.claim())
  );
});

function isHTML(request) {
  return request.mode === 'navigate' || (request.headers.get('accept') || '').includes('text/html');
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Bypass non-GET and SSE
  if (request.method !== 'GET' || url.pathname === '/events') return;

  // Network-first for CSV (avoid stale data like work_orders.csv)
  if (url.pathname.endsWith('.csv')) {
    event.respondWith(
      fetch(request).then((resp) => {
        const copy = resp.clone();
        caches.open(RUNTIME_CACHE).then((c) => c.put(request, copy));
        return resp;
      }).catch(() => caches.match(request))
    );
    return;
  }

  // Network-first for HTML navigations
  if (isHTML(request)) {
    event.respondWith(
      fetch(request).then((resp) => {
        const copy = resp.clone();
        caches.open(RUNTIME_CACHE).then((c) => c.put(request, copy));
        return resp;
      }).catch(() => caches.match(request).then((m) => m || caches.match('/')))
    );
    return;
  }

  // Stale-while-revalidate for same-origin static and CSV
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(request).then((cached) => {
        const fetchPromise = fetch(request).then((resp) => {
          const copy = resp.clone();
          caches.open(RUNTIME_CACHE).then((c) => c.put(request, copy));
          return resp;
        }).catch(() => cached);
        return cached || fetchPromise;
      })
    );
  }
});
