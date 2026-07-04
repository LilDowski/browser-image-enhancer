"""Наглядная демонстрация на НАСТОЯЩИХ фотографиях + независимые (no-reference)
метрики качества. Отвечает на вопрос «как понять, что фото реально стало лучше».

Берём реальные фото, создаём типичные проблемы (недосвет / дымка-низкий контраст /
блёклые цвета), прогоняем через ту же модель, что и в браузере, и показываем:
  - визуальный монтаж  Оригинал | Проблема | Исправлено  -> docs/eval/real_examples.png
  - объективную близость к хорошему оригиналу (PSNR, с эталоном)
  - независимые метрики качества картинки (без эталона):
      * экспозиция  (средняя яркость, хорошо ~0.45–0.55)
      * контраст    (разброс яркости)
      * цветность   (colorfulness по Hasler–Süsstrunk)
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import glob

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = sorted(glob.glob(os.path.join(HERE, "data", "photos", "*.jpg")))
H5 = os.path.join(HERE, "artifacts", "base.h5")
OUT = os.path.join(HERE, "..", "docs", "eval")
os.makedirs(OUT, exist_ok=True)
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def load(path, side=256):
    im = Image.open(path).convert("RGB").resize((side, side))
    return np.asarray(im, np.float32) / 255.0


def apply_params(img, b, c, s):
    x = img + b
    x = (x - 0.5) * c + 0.5
    y = (x * LUMA).sum(-1, keepdims=True)
    x = y + (x - y) * s
    return np.clip(x, 0.0, 1.0)


def model_params(model, img):
    raw = model(img[None], training=False).numpy()[0]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    return float(0.5 * np.tanh(raw[0])), float(0.5 + 1.5 * sig(raw[1])), float(2.0 * sig(raw[2]))


def colorfulness(img):
    """Hasler–Süsstrunk (0..~100): выше = насыщеннее/живее."""
    r, g, b = img[..., 0] * 255, img[..., 1] * 255, img[..., 2] * 255
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def stats(img):
    y = (img * LUMA).sum(-1)
    return y.mean(), y.std(), colorfulness(img)


def psnr(a, b):
    return float(tf.image.psnr(tf.constant(a), tf.constant(b), max_val=1.0))


PROBLEMS = [
    ("Недосвет (тёмное)", (-0.22, 1.0, 1.0)),
    ("Дымка (низкий контраст)", (0.06, 0.6, 0.95)),
    ("Блёклые цвета", (0.0, 1.0, 0.5)),
    ("Тёмное + блёклое", (-0.16, 0.9, 0.7)),
    ("Пересвет", (0.2, 1.0, 1.0)),
    ("Серое и плоское", (0.02, 0.7, 0.65)),
]


def main():
    model = keras.models.load_model(H5)
    rows = []
    print(f"{'Проблема':<26}{'PSNR пробл→ориг':>16}{'PSNR после→ориг':>16}   но-эталон: экспозиция / контраст / цветность")
    for i, (name, dp) in enumerate(PROBLEMS):
        orig = load(PHOTOS[(i * 7 + 3) % len(PHOTOS)])
        problem = apply_params(orig, *dp)
        mb, mc, ms = model_params(model, problem)
        fixed = apply_params(problem, mb, mc, ms)
        rows.append((name, orig, problem, fixed, (mb, mc, ms)))

        p0, p1 = psnr(orig, problem), psnr(orig, fixed)
        ep, cp, fp = stats(problem)
        ef, cf, ff = stats(fixed)
        eo, co, fo = stats(orig)
        print(
            f"{name:<26}{p0:>16.2f}{p1:>16.2f}   "
            f"эксп {ep:.2f}→{ef:.2f} (ориг {eo:.2f}) | "
            f"контр {cp:.2f}→{cf:.2f} | цвет {fp:.0f}→{ff:.0f}"
        )

    # ---- монтаж ----
    cell, pad, lh, head = 220, 10, 24, 34
    cols = 3
    W = cols * cell + (cols + 1) * pad
    H = head + len(rows) * (cell + lh + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 21, 27))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), "Настоящие фото:  Оригинал  |  Типичная проблема  |  Исправлено ИИ", fill=(232, 234, 237))
    titles = ["Оригинал", "Проблема", "Исправлено ИИ"]
    y = head
    for name, orig, problem, fixed, p in rows:
        for j, img in enumerate((orig, problem, fixed)):
            x = pad + j * (cell + pad)
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).resize((cell, cell))
            canvas.paste(im, (x, y))
            cap = f"{name} — {titles[j]}" if j == 0 else titles[j]
            draw.text((x + 4, y + cell + 4), cap, fill=(154, 160, 172))
        y += cell + lh + pad
    out = os.path.join(OUT, "real_examples.png")
    canvas.save(out)
    print("\nМонтаж на реальных фото:", os.path.abspath(out))


if __name__ == "__main__":
    main()
