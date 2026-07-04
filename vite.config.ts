import { defineConfig } from 'vite';

export default defineConfig({
  // Относительные пути — чтобы сборка работала на любом статическом хостинге
  // (GitHub Pages в подкаталоге, Netlify, Vercel и т.д.)
  base: './',
  server: {
    // Привязка к IPv4: иначе Vite слушает только ::1, а часть браузеров идёт на
    // 127.0.0.1 и получает ERR_CONNECTION_REFUSED.
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 4096,
  },
});
