"""Обучение ИИ-модели подбора параметров коррекции (self-supervised).

Идея (см. ТЗ, раздел 7): берём пул хороших фото, случайно портим яркость/контраст/
насыщенность, и учим компактную CNN по «испорченной» версии предсказывать параметры,
которые возвращают изображение к оригиналу. Обучение — через reconstruction loss:
к входу применяется тот же набор операций, что и в браузерном шейдере, и минимизируется
расхождение с оригиналом. Так модель учится «оптимальной» коррекции без ручной разметки.

Результат: компактная Keras-модель → конвертируется в TensorFlow.js (public/model/).

Запуск:  .venv/Scripts/python train.py
"""
import os
import sys
import types

# Keras 2 (через tf-keras) — для совместимости с конвертером tensorflowjs.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Пакет converters в tensorflowjs «жадно» импортирует необязательные зависимости
# (tensorflow_decision_forests и jax), у которых нет колёс под Windows. Для
# конвертации Keras-модели они не используются — подставляем пустые заглушки.
def _stub(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))
    return sys.modules[name]


_stub("tensorflow_decision_forests")
_jax = _stub("jax")
_jax_exp = _stub("jax.experimental")
_jax2tf = _stub("jax.experimental.jax2tf")
_jax.experimental = _jax_exp  # type: ignore[attr-defined]
_jax_exp.jax2tf = _jax2tf  # type: ignore[attr-defined]

import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, "data", "photos")
ART = os.path.join(HERE, "artifacts")
MODEL_OUT = os.path.join(HERE, "..", "public", "model")
os.makedirs(ART, exist_ok=True)
os.makedirs(MODEL_OUT, exist_ok=True)

IMG = 256
BATCH = 16
EPOCHS = 35
STEPS = 60
LUMA = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)


def load_img(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG, IMG])
    return tf.cast(img, tf.float32) / 255.0


def apply_params(img, b, c, s):
    """Та же математика, что в браузерном шейдере: яркость -> контраст -> насыщенность."""
    x = img + b
    x = (x - 0.5) * c + 0.5
    y = tf.reduce_sum(x * LUMA, axis=-1, keepdims=True)
    x = y + (x - y) * s
    return tf.clip_by_value(x, 0.0, 1.0)


def make_pair(img):
    """Возвращает (испорченное, оригинал). Цель обучения — восстановить оригинал.

    В 12% случаев искажения нет (вход = эталон): так модель учится «не навредить»
    уже хорошим фотографиям, но не становится слишком робкой на реально плохих."""
    distort = tf.cast(tf.random.uniform([]) > 0.12, tf.float32)  # 0 => тождество
    b = distort * tf.random.uniform([1, 1, 1], -0.25, 0.25)
    c = 1.0 + distort * (tf.random.uniform([1, 1, 1], 0.6, 1.6) - 1.0)
    s = 1.0 + distort * (tf.random.uniform([1, 1, 1], 0.4, 1.4) - 1.0)
    return apply_params(img, b, c, s), img


def build_base():
    """Компактная CNN: превью 256x256 -> 3 «сырых» числа (логиты параметров).

    Используем обычный Conv2D (а не SeparableConv2D): конвертер tf-keras 2.17
    сериализует SeparableConv2D с лишними полями kernel_*, которые отвергает
    загрузчик TensorFlow.js. Conv2D загружается корректно."""
    inp = keras.Input((IMG, IMG, 3))
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(3)(x)
    return keras.Model(inp, out, name="enhancer")


class ApplyLayer(layers.Layer):
    """Только для обучения: маппит логиты в диапазоны параметров и применяет коррекцию.
    В сохранённую модель НЕ входит (сохраняется лишь base), поэтому модель остаётся
    обычной CNN, максимально совместимой с TensorFlow.js."""

    def call(self, inputs):
        img, r = inputs
        b = 0.5 * tf.tanh(r[:, 0:1])
        c = 0.5 + 1.5 * tf.sigmoid(r[:, 1:2])
        s = 2.0 * tf.sigmoid(r[:, 2:3])
        b = tf.reshape(b, [-1, 1, 1, 1])
        c = tf.reshape(c, [-1, 1, 1, 1])
        s = tf.reshape(s, [-1, 1, 1, 1])
        return apply_params(img, b, c, s)


def dataset(files, training):
    d = tf.data.Dataset.from_tensor_slices(files)
    d = d.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        d = d.map(lambda im: tf.image.random_flip_left_right(im))
        d = d.shuffle(256).repeat()
    d = d.map(make_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return d.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def main():
    files = sorted(glob.glob(os.path.join(PHOTOS, "*.jpg")))
    print(f"Найдено изображений: {len(files)}")
    assert len(files) >= 20, "Слишком мало изображений — сначала запустите download_data.py"

    n_val = max(8, len(files) // 10)
    val_files, train_files = files[:n_val], files[n_val:]
    train_ds, val_ds = dataset(train_files, True), dataset(val_files, False)

    base = build_base()
    base.summary()

    inp = keras.Input((IMG, IMG, 3))
    train_model = keras.Model(inp, ApplyLayer()([inp, base(inp)]))
    train_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS,
        verbose=2,
    )

    h5 = os.path.join(ART, "base.h5")
    base.save(h5)
    print(f"Keras-модель сохранена: {h5}")

    import tensorflowjs as tfjs

    tfjs.converters.save_keras_model(base, MODEL_OUT)
    print(f"Конвертировано в TensorFlow.js: {MODEL_OUT}")


if __name__ == "__main__":
    main()
