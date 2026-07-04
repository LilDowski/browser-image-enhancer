import type {
  WorkerInbound,
  WorkerOutbound,
  CreateTaskOptions,
  TaskStatus,
  EnhanceParams,
  ParamSource,
} from '../api/types';
import { decodeImage } from '../decode/decode';
import { analyze } from '../ml/model';
import { applyWebGL } from '../enhance/webgl';
import { applyCPU } from '../enhance/cpu';

// Множество прерванных задач. Проверяется в контрольных точках пайплайна.
const aborted = new Set<string>();

// Абсолютный URL к model.json (передаётся главным потоком). Пока не задан —
// подбор параметров идёт по классической базовой линии.
let modelBaseUrl: string | undefined;

// Уступаем управление, чтобы worker успел обработать входящие сообщения
// (в частности 'abort') между этапами обработки.
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

class AbortError extends Error {
  constructor() {
    super('aborted');
  }
}

function post(msg: WorkerOutbound, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

function emit(
  taskId: string,
  status: TaskStatus,
  progress: number,
  params?: EnhanceParams,
  message?: string,
  source?: ParamSource,
) {
  post({ type: 'progress', taskId, status, progress, message, params, source });
}

function checkAbort(taskId: string) {
  if (aborted.has(taskId)) throw new AbortError();
}

/** Ограничивает размер изображения (защита от чрезмерного потребления памяти). */
async function clampSize(bitmap: ImageBitmap, maxDim: number): Promise<ImageBitmap> {
  const longest = Math.max(bitmap.width, bitmap.height);
  if (!maxDim || longest <= maxDim) return bitmap;
  const scale = maxDim / longest;
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const c = new OffscreenCanvas(w, h);
  c.getContext('2d')!.drawImage(bitmap, 0, 0, w, h);
  return createImageBitmap(c);
}

/** Уменьшенная копия для анализа моделью. */
function makeThumb(bitmap: ImageBitmap, size = 256): ImageData {
  const c = new OffscreenCanvas(size, size);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

async function process(
  taskId: string,
  buffer: ArrayBuffer,
  mime: string,
  options: CreateTaskOptions,
) {
  const maxDim = options.maxDimension ?? 8192;
  const outputType = options.outputType ?? 'image/jpeg';
  const quality = options.quality ?? 0.92;

  // 1. Декодирование
  emit(taskId, 'decoding', 5, undefined, 'Декодирование изображения');
  await tick();
  checkAbort(taskId);
  let bitmap = await decodeImage(buffer, mime);
  bitmap = await clampSize(bitmap, maxDim);
  emit(taskId, 'decoding', 25);
  await tick();
  checkAbort(taskId);

  // 2. Анализ — подбор параметров (ИИ / базовая линия)
  emit(taskId, 'analyzing', 45, undefined, 'Подбор параметров');
  const thumb = makeThumb(bitmap);
  await tick();
  checkAbort(taskId);
  const { params, source, diag } = await analyze(thumb, modelBaseUrl);
  const srcMsg =
    source === 'model'
      ? 'Параметры подобраны ИИ-моделью'
      : `Базовый алгоритм${diag ? ` (${diag})` : ''}`;
  emit(taskId, 'analyzing', 60, params, srcMsg, source);
  await tick();
  checkAbort(taskId);

  // 3. Применение коррекции к полному кадру
  emit(taskId, 'enhancing', 70, params, 'Применение коррекции', source);
  await tick();
  checkAbort(taskId);
  const canvas = applyWebGL(bitmap, params) ?? applyCPU(bitmap, params);
  emit(taskId, 'enhancing', 90, params, undefined, source);
  await tick();
  checkAbort(taskId);

  // 4. Кодирование результата
  const blob = await canvas.convertToBlob({ type: outputType, quality });
  post({
    type: 'done',
    taskId,
    blob,
    width: bitmap.width,
    height: bitmap.height,
    params,
    source,
  });
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    modelBaseUrl = msg.modelBaseUrl;
    return;
  }
  if (msg.type === 'abort') {
    aborted.add(msg.taskId);
    return;
  }
  if (msg.type === 'process') {
    try {
      await process(msg.taskId, msg.buffer, msg.mime, msg.options);
    } catch (err) {
      if (err instanceof AbortError) {
        post({ type: 'aborted', taskId: msg.taskId });
      } else {
        post({
          type: 'error',
          taskId: msg.taskId,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    } finally {
      aborted.delete(msg.taskId);
    }
  }
};
