/**
 * Определяет, является ли буфер изображением HEIC/HEIF — по MIME или по
 * сигнатуре ftyp-бренда в начале файла.
 */
function isHeic(buffer: ArrayBuffer, mime: string): boolean {
  if (/hei[cf]/i.test(mime)) return true;
  if (buffer.byteLength < 12) return false;
  const b = new Uint8Array(buffer, 0, 12);
  // байты 4..8 = 'ftyp', далее бренд
  const brand = String.fromCharCode(b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11]);
  return /ftyp(heic|heix|hevc|hevx|mif1|heim|heis|hevm)/i.test(brand);
}

/**
 * Декодирует произвольное изображение в ImageBitmap.
 * JPG/PNG/BMP — нативно через createImageBitmap.
 * HEIC — через WASM-декодер, который подгружается лениво (только при необходимости).
 */
export async function decodeImage(buffer: ArrayBuffer, mime: string): Promise<ImageBitmap> {
  if (isHeic(buffer, mime)) {
    const { decodeHeic } = await import('./heic');
    return decodeHeic(buffer);
  }
  const blob = new Blob([buffer], { type: mime || 'application/octet-stream' });
  return createImageBitmap(blob);
}
