import type { EnhanceParams, ParamSource } from '../api/types';
import { analyzeBaseline } from '../analyze/baseline';

// Слой подбора параметров. Пытается загрузить обученную ML-модель (TensorFlow.js);
// если модели нет — прозрачно откатывается на классическую базовую линию.
// TF.js импортируется динамически, поэтому НЕ попадает в загрузку, пока модель
// фактически не появилась в public/model/.

type LoadedModel = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tf: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  net: any;
} | null;

export interface AnalyzeResult {
  params: EnhanceParams;
  source: ParamSource;
  /** Диагностика причины отката на базовую линию (для отладки). */
  diag?: string;
}

let modelTried = false;
let modelPromise: Promise<LoadedModel> | null = null;
let lastDiag: string | undefined;

async function loadModel(url: string): Promise<LoadedModel> {
  if (modelTried) return modelPromise;
  modelTried = true;
  modelPromise = (async () => {
    try {
      const probe = await fetch(url, { method: 'GET' });
      if (!probe.ok) {
        lastDiag = `probe ${probe.status}`;
        return null; // модели нет — используем базовую линию
      }
      const tf = await import('@tensorflow/tfjs');
      // Модель сохраняется конвертером как Keras layers-модель.
      const net = await tf.loadLayersModel(url);
      return { tf, net };
    } catch (err) {
      lastDiag = 'load: ' + (err instanceof Error ? err.message : String(err));
      return null;
    }
  })();
  return modelPromise;
}

/**
 * Подбор параметров коррекции по превью (модель → базовая линия).
 * @param modelUrl абсолютный URL к model.json; если не задан — сразу базовая линия.
 */
export async function analyze(thumb: ImageData, modelUrl?: string): Promise<AnalyzeResult> {
  lastDiag = modelUrl ? undefined : 'no modelUrl';
  const model = modelUrl ? await loadModel(modelUrl) : null;
  if (model && model.net) {
    try {
      const { tf, net } = model;
      const raw: number[] = tf.tidy(() => {
        const x = tf.browser.fromPixels(thumb).toFloat().div(255).expandDims(0);
        const y = net.predict(x);
        return Array.from(y.dataSync() as Float32Array);
      });
      // Маппинг «сырых» логитов в диапазоны параметров — должен совпадать
      // с ApplyLayer в training/train.py.
      const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
      return {
        params: {
          brightness: 0.5 * Math.tanh(raw[0]),
          contrast: 0.5 + 1.5 * sigmoid(raw[1]),
          saturation: 2.0 * sigmoid(raw[2]),
        },
        source: 'model',
      };
    } catch (err) {
      lastDiag = 'predict: ' + (err instanceof Error ? err.message : String(err));
    }
  }
  return { params: analyzeBaseline(thumb), source: 'baseline', diag: lastDiag };
}
