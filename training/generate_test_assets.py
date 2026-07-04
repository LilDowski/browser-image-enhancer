"""Готовит набор реальных тестовых файлов для проверки системы по ТЗ.

Из скачанных фото собирает:
  - photo_15mp.jpg  — мозаика ~15 Мпк (проверка лимита разрешения и скорости)
  - sample.png      — PNG
  - sample.bmp      — BMP
  - sample.heic     — HEIC (через pillow-heif)
  - portrait.jpg    — вертикальное фото (проверка ориентации EXIF/портрет)

Файлы кладём в public/test-assets/ (Vite раздаёт их по URL). Перед деплоем папка
удаляется — в прод-сборку не идёт.
"""
import glob
import os

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = sorted(glob.glob(os.path.join(HERE, "data", "photos", "*.jpg")))
OUT = os.path.join(HERE, "..", "public", "test-assets")
os.makedirs(OUT, exist_ok=True)

assert PHOTOS, "Нет фото — сначала download_data.py"


def load(i):
    return Image.open(PHOTOS[i % len(PHOTOS)]).convert("RGB")


def mosaic(width, height, tile=320):
    canvas = Image.new("RGB", (width, height))
    idx = 0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            canvas.paste(load(idx).resize((tile, tile)), (x, y))
            idx += 1
    return canvas


# 15 Мпк: 4472 x 3354 = 14 999 088 ≈ 15.0 Мпк
big = mosaic(4472, 3354)
big.save(os.path.join(OUT, "photo_15mp.jpg"), quality=90)
print("photo_15mp.jpg", big.size, round(big.size[0] * big.size[1] / 1e6, 2), "Мпк")

# Средние форматы из одного реального снимка
mid = mosaic(1600, 1200)
mid.save(os.path.join(OUT, "sample.png"))
mid.save(os.path.join(OUT, "sample.bmp"))
mid.save(os.path.join(OUT, "sample.heic"), quality=88)
print("sample.png / sample.bmp / sample.heic", mid.size)

# Портретное (вертикальное) — проверка, что ориентация не путается
portrait = mosaic(1200, 1800)
portrait.save(os.path.join(OUT, "portrait.jpg"), quality=90)
print("portrait.jpg", portrait.size)

print("Готово ->", os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f:18} {os.path.getsize(p)/1024:8.1f} КБ")
