import type {
  CreateTaskOptions,
  EnhanceParams,
  ParamSource,
  TaskProgress,
  TaskStatus,
  WorkerOutbound,
} from './types';

interface TaskRecord {
  taskId: string;
  status: TaskStatus;
  progress: number;
  message?: string;
  params?: EnhanceParams;
  source?: ParamSource;
  result?: Blob;
  width?: number;
  height?: number;
  error?: string;
}

const FINAL: TaskStatus[] = ['done', 'aborted', 'error'];
const isFinal = (s: TaskStatus) => FINAL.includes(s);

async function toBuffer(
  input: File | Blob | ArrayBuffer | ImageBitmap,
): Promise<{ buffer: ArrayBuffer; mime: string }> {
  if (input instanceof Blob) {
    return { buffer: await input.arrayBuffer(), mime: input.type };
  }
  if (input instanceof ArrayBuffer) {
    return { buffer: input, mime: '' };
  }
  if (typeof ImageBitmap !== 'undefined' && input instanceof ImageBitmap) {
    const c =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(input.width, input.height)
        : Object.assign(document.createElement('canvas'), {
            width: input.width,
            height: input.height,
          });
    (c.getContext('2d') as CanvasRenderingContext2D).drawImage(input, 0, 0);
    const blob =
      c instanceof OffscreenCanvas
        ? await c.convertToBlob({ type: 'image/png' })
        : await new Promise<Blob>((res) =>
            (c as HTMLCanvasElement).toBlob((b) => res(b!), 'image/png'),
          );
    return { buffer: await blob.arrayBuffer(), mime: 'image/png' };
  }
  throw new Error('Неподдерживаемый тип входных данных');
}

function blobToDataURL(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as string);
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(blob);
  });
}

/**
 * Публичный API системы улучшения изображений.
 *
 * Наследует EventTarget — изменения статуса приходят событием 'statuschange'
 * с CustomEvent<TaskProgress> в свойстве detail.
 *
 * @example
 * const enhancer = new ImageEnhancer();
 * enhancer.addEventListener('statuschange', (e) => {
 *   const { status, progress } = (e as CustomEvent<TaskProgress>).detail;
 * });
 * const id = await enhancer.createTask(file);
 * const blob = await enhancer.getResult(id);
 */
export class ImageEnhancer extends EventTarget {
  private worker: Worker;
  private tasks = new Map<string, TaskRecord>();

  constructor() {
    super();
    this.worker = new Worker(new URL('../worker/enhance.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (e: MessageEvent<WorkerOutbound>) => this.onMessage(e.data);

    // Абсолютный URL модели вычисляем от базы документа — работает и в dev,
    // и при деплое в подкаталог (GitHub Pages и т.п.).
    const modelBaseUrl = new URL('model/model.json', document.baseURI).href;
    this.worker.postMessage({ type: 'init', modelBaseUrl });
  }

  /** Метод постановки задачи — возвращает идентификатор задачи. */
  async createTask(
    input: File | Blob | ArrayBuffer | ImageBitmap,
    options: CreateTaskOptions = {},
  ): Promise<string> {
    const taskId =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `task-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    this.tasks.set(taskId, { taskId, status: 'queued', progress: 0 });
    this.update(taskId, 'queued', 0, 'В очереди');

    const { buffer, mime } = await toBuffer(input);
    this.worker.postMessage({ type: 'process', taskId, buffer, mime, options }, [buffer]);
    return taskId;
  }

  /** Метод получения статуса задачи — текущий статус и прогресс. */
  getStatus(taskId: string): TaskProgress {
    const t = this.tasks.get(taskId);
    if (!t) throw new Error(`Задача не найдена: ${taskId}`);
    return {
      taskId: t.taskId,
      status: t.status,
      progress: t.progress,
      message: t.message,
      params: t.params,
      source: t.source,
      error: t.error,
    };
  }

  /** Метод прерывания задачи — возвращает информацию об успешности. */
  async abortTask(taskId: string): Promise<{ success: boolean }> {
    const t = this.tasks.get(taskId);
    if (!t || isFinal(t.status)) return { success: false };
    this.worker.postMessage({ type: 'abort', taskId });
    return { success: true };
  }

  /** Метод получения готового изображения. */
  async getResult(
    taskId: string,
    as: 'blob' | 'bitmap' | 'dataurl' = 'blob',
  ): Promise<Blob | ImageBitmap | string> {
    const t = this.tasks.get(taskId);
    if (!t) throw new Error(`Задача не найдена: ${taskId}`);
    if (t.status !== 'done' || !t.result) {
      throw new Error(`Результат не готов (статус: ${t.status})`);
    }
    if (as === 'blob') return t.result;
    if (as === 'bitmap') return createImageBitmap(t.result);
    return blobToDataURL(t.result);
  }

  /** Освобождает ресурсы worker'а. */
  dispose() {
    this.worker.terminate();
    this.tasks.clear();
  }

  private onMessage(msg: WorkerOutbound) {
    const t = this.tasks.get(msg.taskId);
    if (!t) return;

    switch (msg.type) {
      case 'progress':
        if (msg.params) t.params = msg.params;
        if (msg.source) t.source = msg.source;
        this.update(msg.taskId, msg.status, msg.progress, msg.message);
        break;
      case 'done':
        t.result = msg.blob;
        t.params = msg.params;
        t.source = msg.source;
        t.width = msg.width;
        t.height = msg.height;
        this.update(msg.taskId, 'done', 100, 'Готово');
        break;
      case 'aborted':
        this.update(msg.taskId, 'aborted', t.progress, 'Прервано');
        break;
      case 'error':
        t.error = msg.error;
        this.update(msg.taskId, 'error', t.progress, 'Ошибка');
        break;
    }
  }

  private update(taskId: string, status: TaskStatus, progress: number, message?: string) {
    const t = this.tasks.get(taskId);
    if (!t) return;
    t.status = status;
    t.progress = progress;
    t.message = message;

    const detail: TaskProgress = {
      taskId,
      status,
      progress,
      message,
      params: t.params,
      source: t.source,
      error: t.error,
    };
    this.dispatchEvent(new CustomEvent<TaskProgress>('statuschange', { detail }));
  }
}
