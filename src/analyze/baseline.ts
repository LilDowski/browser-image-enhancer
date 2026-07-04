import type { EnhanceParams } from '../api/types';

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/**
 * Классический авто-алгоритм подбора параметров (базовая линия).
 *
 * Анализирует уменьшенную копию изображения (превью) и подбирает яркость,
 * контраст и насыщенность так, чтобы привести статистику изображения к
 * «приятным» целевым значениям. Используется, пока не подключена ML-модель,
 * и служит точкой сравнения качества (см. ТЗ, раздел 7.5).
 */
export function analyzeBaseline(thumb: ImageData): EnhanceParams {
  const data = thumb.data;
  const n = data.length / 4;

  let sumY = 0;
  let sumY2 = 0;
  let sumSat = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;

    const y = 0.299 * r + 0.587 * g + 0.114 * b; // яркость (luma)
    sumY += y;
    sumY2 += y * y;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const sat = max <= 0 ? 0 : (max - min) / max; // насыщенность (HSV)
    sumSat += sat;
  }

  const meanY = sumY / n;
  const varY = Math.max(0, sumY2 / n - meanY * meanY);
  const stdY = Math.sqrt(varY);
  const meanSat = sumSat / n;

  // Яркость: частично подтягиваем среднюю яркость к 0.5.
  const targetY = 0.5;
  const brightness = clamp((targetY - meanY) * 0.6, -0.4, 0.4);

  // Контраст: масштабируем разброс яркости к целевому std.
  const targetStd = 0.22;
  const contrast = clamp(stdY > 0.02 ? targetStd / stdY : 1, 0.8, 1.8);

  // Насыщенность: оживляем блёклые изображения, не пересыщая яркие.
  const targetSat = 0.4;
  const saturation = clamp(meanSat > 0.02 ? targetSat / meanSat : 1, 0.8, 1.6);

  return { brightness, contrast, saturation };
}
