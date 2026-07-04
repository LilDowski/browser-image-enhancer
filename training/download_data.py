"""Скачивает пул реальных фотографий для обучения модели.

Использует picsum.photos (фото из Unsplash, открытая лицензия). Только stdlib —
не требует установки зависимостей, поэтому запускается системным Python сразу,
параллельно с установкой тяжёлого ML-окружения.

Запуск:  python download_data.py [N]   (по умолчанию N=300)
"""
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "photos")
os.makedirs(OUT, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
W = H = 320


def fetch(seed: int) -> bytes:
    url = f"https://picsum.photos/seed/vk{seed}/{W}/{H}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read()


def main() -> None:
    ok = 0
    for i in range(N):
        path = os.path.join(OUT, f"img_{i:04d}.jpg")
        if os.path.exists(path) and os.path.getsize(path) > 1500:
            ok += 1
            continue
        for attempt in range(3):
            try:
                data = fetch(i)
                if len(data) > 1500:
                    with open(path, "wb") as f:
                        f.write(data)
                    ok += 1
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"skip {i}: {e}", flush=True)
                else:
                    time.sleep(1.0)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{N} processed, ok={ok}", flush=True)
    print(f"DONE ok={ok} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
