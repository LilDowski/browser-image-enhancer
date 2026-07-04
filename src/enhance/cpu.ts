import type { EnhanceParams } from '../api/types';

const clamp255 = (x: number) => (x < 0 ? 0 : x > 255 ? 255 : x);

/**
 * CPU-фоллбэк применения коррекции (если WebGL/OffscreenCanvas недоступны).
 * Та же математика, что в шейдере, но над Uint8ClampedArray.
 */
export function applyCPU(source: ImageBitmap, p: EnhanceParams): OffscreenCanvas {
  const canvas = new OffscreenCanvas(source.width, source.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(source, 0, 0);

  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = img.data;
  const { brightness, contrast, saturation } = p;

  for (let i = 0; i < d.length; i += 4) {
    let r = d[i] / 255;
    let g = d[i + 1] / 255;
    let b = d[i + 2] / 255;

    // Яркость
    r += brightness;
    g += brightness;
    b += brightness;

    // Контраст
    r = (r - 0.5) * contrast + 0.5;
    g = (g - 0.5) * contrast + 0.5;
    b = (b - 0.5) * contrast + 0.5;

    // Насыщенность
    const Y = 0.299 * r + 0.587 * g + 0.114 * b;
    r = Y + (r - Y) * saturation;
    g = Y + (g - Y) * saturation;
    b = Y + (b - Y) * saturation;

    d[i] = clamp255(r * 255);
    d[i + 1] = clamp255(g * 255);
    d[i + 2] = clamp255(b * 255);
  }

  ctx.putImageData(img, 0, 0);
  return canvas;
}
