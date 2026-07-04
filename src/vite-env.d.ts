/// <reference types="vite/client" />

// У пакета libheif-js нет собственных типов — объявляем модуль вручную.
declare module 'libheif-js/wasm-bundle';
declare module 'libheif-js';
