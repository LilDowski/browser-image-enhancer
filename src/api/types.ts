// Публичные типы API системы улучшения изображений.
// Соответствуют спецификации из docs/TECHNICAL_SPEC.md (раздел 5).

export type TaskStatus =
  | 'queued'    // поставлена в очередь
  | 'decoding'  // декодирование исходника
  | 'analyzing' // подбор параметров (инференс / базовая линия)
  | 'enhancing' // применение коррекции
  | 'done'      // готово
  | 'aborted'   // прервана
  | 'error';    // ошибка

/** Источник подобранных параметров: ML-модель или классическая базовая линия. */
export type ParamSource = 'model' | 'baseline';

/** Параметры коррекции, которые подбирает ИИ и применяет алгоритм. */
export interface EnhanceParams {
  /** Сдвиг яркости, диапазон ~[-0.5 .. +0.5] */
  brightness: number;
  /** Множитель контраста, диапазон ~[0.5 .. 2.0] */
  contrast: number;
  /** Множитель насыщенности, диапазон ~[0.0 .. 2.0] */
  saturation: number;
}

/** Текущее состояние задачи — то, что возвращает getStatus() и приходит в событии. */
export interface TaskProgress {
  taskId: string;
  status: TaskStatus;
  /** Прогресс выполнения, 0..100 */
  progress: number;
  /** Человекочитаемое описание текущего этапа */
  message?: string;
  /** Подобранные параметры (доступны начиная с этапа analyzing) */
  params?: EnhanceParams;
  /** Чем подобраны параметры — моделью или базовой линией */
  source?: ParamSource;
  /** Текст ошибки (для status === 'error') */
  error?: string;
}

export interface CreateTaskOptions {
  /** Формат выходного изображения */
  outputType?: 'image/jpeg' | 'image/png';
  /** Качество JPEG, 0..1 */
  quality?: number;
  /** Ограничение большей стороны (защита от OOM). По умолчанию 8192. */
  maxDimension?: number;
}

// --- Внутренний протокол обмена с Web Worker ---

export type WorkerInbound =
  | { type: 'init'; modelBaseUrl: string }
  | {
      type: 'process';
      taskId: string;
      buffer: ArrayBuffer;
      mime: string;
      options: CreateTaskOptions;
    }
  | { type: 'abort'; taskId: string };

export type WorkerOutbound =
  | {
      type: 'progress';
      taskId: string;
      status: TaskStatus;
      progress: number;
      message?: string;
      params?: EnhanceParams;
      source?: ParamSource;
    }
  | {
      type: 'done';
      taskId: string;
      blob: Blob;
      width: number;
      height: number;
      params: EnhanceParams;
      source: ParamSource;
    }
  | { type: 'aborted'; taskId: string }
  | { type: 'error'; taskId: string; error: string };
