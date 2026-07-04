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
    rollupOptions: {
      output: {
        // Чанки worker-сборки (TF.js, heic, _commonjsHelpers) тоже не должны
        // начинаться с '_' — см. комментарий в build.rollupOptions ниже.
        chunkFileNames: 'assets/chunk-[name]-[hash].js',
      },
    },
  },
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 4096,
    rollupOptions: {
      output: {
        // Префикс 'chunk-' гарантирует, что имя файла не начнётся с '_'
        // (файлы вида _commonjsHelpers-*.js Jekyll на GitHub Pages отдаёт как 404).
        chunkFileNames: 'assets/chunk-[name]-[hash].js',
      },
    },
  },
});
