// HEIC-декодер на базе libheif (WASM). Импортируется динамически из decode.ts,
// поэтому ~2 МБ WASM попадают в загрузку только когда пользователь действительно
// открыл HEIC-файл, а не при старте приложения.
import libheif from 'libheif-js/wasm-bundle';

export async function decodeHeic(buffer: ArrayBuffer): Promise<ImageBitmap> {
  const decoder = new libheif.HeifDecoder();
  const images = decoder.decode(new Uint8Array(buffer));
  if (!images || images.length === 0) {
    throw new Error('HEIC: не удалось извлечь кадры');
  }

  const image = images[0];
  const width = image.get_width();
  const height = image.get_height();
  const imageData = new ImageData(width, height);

  await new Promise<void>((resolve, reject) => {
    image.display(imageData, (displayData: ImageData | null) => {
      if (!displayData) reject(new Error('HEIC: ошибка декодирования кадра'));
      else resolve();
    });
  });

  return createImageBitmap(imageData);
}
