"""Оценка качества работы системы на эталонном пуле (ТЗ, этапы 3 и 6).

Методика «degrade-restore»:
  1. Берём свежие эталонные фото, которых модель НЕ видела при обучении.
  2. Намеренно портим их с известными параметрами по категориям
     (тёмное / пересвет / низкий контраст / блёклый цвет / смешанное).
  3. Прогоняем испорченное через систему и сравниваем результат с оригиналом
     метриками PSNR (дБ) и SSIM.
  4. Сравниваем три стратегии: без коррекции / ИИ-модель / классическая база.

Применяемая математика коррекции идентична браузерному WebGL-шейдеру
(src/enhance/webgl.ts) и базовой линии (src/analyze/baseline.ts), поэтому
числа репрезентативны для реального решения.

Запуск:  .venv/Scripts/python eval.py
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import glob
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "data", "eval_photos")
DOCS_EVAL = os.path.join(HERE, "..", "docs", "eval")
H5 = os.path.join(HERE, "artifacts", "base.h5")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(DOCS_EVAL, exist_ok=True)

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
N_EVAL = 80
MAX_SIDE = 512
RNG = np.random.default_rng(20260630)


# ---------- данные ----------
def download_eval(n=N_EVAL):
    ok = 0
    for i in range(n):
        path = os.path.join(EVAL_DIR, f"eval_{i:03d}.jpg")
        if os.path.exists(path) and os.path.getsize(path) > 1500:
            ok += 1
            continue
        url = f"https://picsum.photos/seed/holdout{i}/320/320"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if len(data) > 1500:
                with open(path, "wb") as f:
                    f.write(data)
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"skip {i}: {e}", flush=True)
    print(f"Эталонный пул: {ok} изображений в {EVAL_DIR}")


def load_rgb(path, max_side=MAX_SIDE):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((round(w * scale), round(h * scale)))
    return np.asarray(im, np.float32) / 255.0


# ---------- коррекция (как в шейдере) ----------
def apply_params(img, b, c, s):
    x = img + b
    x = (x - 0.5) * c + 0.5
    y = (x * LUMA).sum(-1, keepdims=True)
    x = y + (x - y) * s
    return np.clip(x, 0.0, 1.0)


def thumb256(img):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).resize((256, 256))
    return np.asarray(im, np.float32) / 255.0


# ---------- подбор параметров ----------
def model_params(model, img):
    t = thumb256(img)[None]
    raw = model(t, training=False).numpy()[0]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    return float(0.5 * np.tanh(raw[0])), float(0.5 + 1.5 * sig(raw[1])), float(2.0 * sig(raw[2]))


def baseline_params(img):
    """Точная копия analyzeBaseline из src/analyze/baseline.ts."""
    t = thumb256(img)
    y = (t * LUMA).sum(-1)
    mean_y, std_y = float(y.mean()), float(y.std())
    mx, mn = t.max(-1), t.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    mean_sat = float(sat.mean())
    b = np.clip((0.5 - mean_y) * 0.6, -0.4, 0.4)
    c = np.clip(0.22 / std_y if std_y > 0.02 else 1.0, 0.8, 1.8)
    s = np.clip(0.4 / mean_sat if mean_sat > 0.02 else 1.0, 0.8, 1.6)
    return float(b), float(c), float(s)


# ---------- метрики ----------
def metrics(a, b):
    ta, tb = tf.constant(a), tf.constant(b)
    psnr = float(tf.image.psnr(ta, tb, max_val=1.0))
    ssim = float(tf.image.ssim(ta, tb, max_val=1.0))
    return psnr, ssim


# ---------- деградации ----------
def make_degradation(category):
    u = RNG.uniform
    if category == "Тёмное":
        return (u(-0.28, -0.16), 1.0, 1.0)
    if category == "Пересвет":
        return (u(0.16, 0.26), 1.0, 1.0)
    if category == "Низкий контраст":
        return (u(-0.04, 0.04), u(0.58, 0.74), 1.0)
    if category == "Блёклый цвет":
        return (0.0, 1.0, u(0.40, 0.62))
    # Смешанное
    return (u(-0.18, 0.18), u(0.7, 1.25), u(0.55, 1.3))


CATEGORIES = ["Тёмное", "Пересвет", "Низкий контраст", "Блёклый цвет", "Смешанное"]


# ---------- монтаж примеров ----------
def save_montage(model, files):
    rows = []
    for cat, path in zip(CATEGORIES, files):
        orig = load_rgb(path, 256)
        h, w = orig.shape[:2]
        side = min(h, w)
        orig = orig[:side, :side]  # квадрат для аккуратной сетки
        db, dc, ds = make_degradation(cat)
        degraded = apply_params(orig, db, dc, ds)
        mb, mc, ms = model_params(model, degraded)
        restored = apply_params(degraded, mb, mc, ms)
        rows.append((cat, orig, degraded, restored))

    cell = 220
    pad = 10
    label_h = 26
    cols = 3
    head = 34
    W = cols * cell + (cols + 1) * pad
    H = head + len(rows) * (cell + label_h + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 21, 27))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), "Оригинал  |  Испорчено  |  Восстановлено ИИ", fill=(232, 234, 237))

    def put(img, x, y, title):
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).resize((cell, cell))
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + cell + 4), title, fill=(154, 160, 172))

    titles = ["Оригинал", "Испорчено", "Восстановлено"]
    y = head
    for cat, orig, degraded, restored in rows:
        for j, img in enumerate((orig, degraded, restored)):
            x = pad + j * (cell + pad)
            put(img, x, y, f"{cat} — {titles[j]}" if j == 0 else titles[j])
        y += cell + label_h + pad

    out = os.path.join(DOCS_EVAL, "examples.png")
    canvas.save(out)
    print("Монтаж примеров:", os.path.abspath(out))


# ---------- основной прогон ----------
def main():
    download_eval()
    files = sorted(glob.glob(os.path.join(EVAL_DIR, "*.jpg")))
    assert files, "Нет эталонных изображений"
    print(f"Загрузка модели: {H5}")
    model = keras.models.load_model(H5)

    acc = {c: {"degraded": [], "model": [], "baseline": []} for c in CATEGORIES}
    acc_ssim = {c: {"degraded": [], "model": [], "baseline": []} for c in CATEGORIES}

    for path in files:
        orig = load_rgb(path)
        for cat in CATEGORIES:
            db, dc, ds = make_degradation(cat)
            degraded = apply_params(orig, db, dc, ds)

            mb, mc, ms = model_params(model, degraded)
            restored_m = apply_params(degraded, mb, mc, ms)

            bb, bc, bs = baseline_params(degraded)
            restored_b = apply_params(degraded, bb, bc, bs)

            for key, img in (("degraded", degraded), ("model", restored_m), ("baseline", restored_b)):
                p, s = metrics(orig, img)
                acc[cat][key].append(p)
                acc_ssim[cat][key].append(s)

    def mean(xs):
        return float(np.mean(xs))

    print("\n================ КАЧЕСТВО (PSNR, дБ — больше лучше) ================")
    print(f"{'Категория':<18}{'Без корр.':>11}{'ИИ-модель':>12}{'База':>10}{'Прирост ИИ':>13}")
    overall = {"degraded": [], "model": [], "baseline": []}
    overall_ssim = {"degraded": [], "model": [], "baseline": []}
    for cat in CATEGORIES:
        d, m, b = mean(acc[cat]["degraded"]), mean(acc[cat]["model"]), mean(acc[cat]["baseline"])
        print(f"{cat:<18}{d:>11.2f}{m:>12.2f}{b:>10.2f}{m - d:>+12.2f}")
        for k in overall:
            overall[k] += acc[cat][k]
            overall_ssim[k] += acc_ssim[cat][k]
    od, om, ob = mean(overall["degraded"]), mean(overall["model"]), mean(overall["baseline"])
    print("-" * 64)
    print(f"{'ИТОГО':<18}{od:>11.2f}{om:>12.2f}{ob:>10.2f}{om - od:>+12.2f}")

    print("\n================ КАЧЕСТВО (SSIM — ближе к 1 лучше) ================")
    print(f"{'Категория':<18}{'Без корр.':>11}{'ИИ-модель':>12}{'База':>10}")
    for cat in CATEGORIES:
        d = mean(acc_ssim[cat]["degraded"])
        m = mean(acc_ssim[cat]["model"])
        b = mean(acc_ssim[cat]["baseline"])
        print(f"{cat:<18}{d:>11.3f}{m:>12.3f}{b:>10.3f}")
    sd, sm, sb = mean(overall_ssim["degraded"]), mean(overall_ssim["model"]), mean(overall_ssim["baseline"])
    print("-" * 52)
    print(f"{'ИТОГО':<18}{sd:>11.3f}{sm:>12.3f}{sb:>10.3f}")

    win = np.mean(np.array(overall["model"]) > np.array(overall["degraded"])) * 100
    print(f"\nДоля случаев, где ИИ улучшил изображение: {win:.1f}%")

    # ---- Тест «не навреди»: неискажённые хорошие фото должны почти не меняться ----
    harm = {"model": [], "baseline": []}
    harm_ssim = {"model": [], "baseline": []}
    dev = {"model": [], "baseline": []}
    for path in files:
        orig = load_rgb(path)
        for key, params in (("model", model_params(model, orig)), ("baseline", baseline_params(orig))):
            pb, pc, ps = params
            restored = apply_params(orig, pb, pc, ps)
            p, s = metrics(orig, restored)
            harm[key].append(p)
            harm_ssim[key].append(s)
            dev[key].append(abs(pb) + abs(pc - 1.0) + abs(ps - 1.0))
    print("\n============ ТЕСТ «НЕ НАВРЕДИ» (хорошие фото, чем выше — тем меньше вреда) ============")
    print(f"{'Стратегия':<14}{'PSNR, дБ':>12}{'SSIM':>10}{'Сила правки':>14}")
    print(f"{'ИИ-модель':<14}{mean(harm['model']):>12.2f}{mean(harm_ssim['model']):>10.3f}{mean(dev['model']):>14.3f}")
    print(f"{'Классич. база':<14}{mean(harm['baseline']):>12.2f}{mean(harm_ssim['baseline']):>10.3f}{mean(dev['baseline']):>14.3f}")

    summary = {
        "images": len(files),
        "cases": len(files) * len(CATEGORIES),
        "psnr": {"degraded": od, "model": om, "baseline": ob},
        "ssim": {"degraded": sd, "model": sm, "baseline": sb},
        "improvement_rate_percent": float(win),
        "do_no_harm": {
            "model": {"psnr": mean(harm["model"]), "ssim": mean(harm_ssim["model"]), "edit_strength": mean(dev["model"])},
            "baseline": {"psnr": mean(harm["baseline"]), "ssim": mean(harm_ssim["baseline"]), "edit_strength": mean(dev["baseline"])},
        },
        "per_category_psnr": {
            c: {
                "degraded": mean(acc[c]["degraded"]),
                "model": mean(acc[c]["model"]),
                "baseline": mean(acc[c]["baseline"]),
            }
            for c in CATEGORIES
        },
    }
    with open(os.path.join(DOCS_EVAL, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nМетрики сохранены:", os.path.abspath(os.path.join(DOCS_EVAL, "metrics.json")))

    save_montage(model, files[:5])


if __name__ == "__main__":
    main()
