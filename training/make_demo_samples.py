"""Готовит демо-примеры для кнопок «Попробовать на примере» в UI.

Скачивает 3 реальных фото в среднем разрешении и создаёт из них типичные
проблемные кадры (тёмное / дымка / блёклое). Кладёт в public/samples/ —
эти файлы ЕДУТ в прод-сборку (в отличие от test-assets), поэтому размер
держим скромным (~1280px, JPEG q83).
"""
import io
import os
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "samples")
os.makedirs(OUT, exist_ok=True)
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def fetch(seed, w=1280, h=860):
    url = f"https://picsum.photos/seed/{seed}/{w}/{h}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def apply(img, b, c, s):
    x = np.asarray(img, np.float32) / 255.0
    x = (x + b - 0.5) * c + 0.5
    y = (x * LUMA).sum(-1, keepdims=True)
    x = np.clip(y + (x - y) * s, 0, 1)
    return Image.fromarray((x * 255).astype("uint8"))


SAMPLES = [
    ("dark.jpg", "demo-dark-7", (-0.22, 0.95, 0.9)),   # недосвет
    ("haze.jpg", "demo-haze-4", (0.07, 0.58, 0.85)),   # дымка / низкий контраст
    ("dull.jpg", "demo-dull-2", (-0.04, 0.85, 0.45)),  # блёклые цвета
]

for fname, seed, (b, c, s) in SAMPLES:
    img = apply(fetch(seed), b, c, s)
    path = os.path.join(OUT, fname)
    img.save(path, quality=83)
    print(f"{fname}: {img.size}, {os.path.getsize(path)/1024:.0f} КБ")

print("Готово ->", os.path.abspath(OUT))
