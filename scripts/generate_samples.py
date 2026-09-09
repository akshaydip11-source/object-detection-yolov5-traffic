#!/usr/bin/env python3
"""Generate synthetic traffic-like sample images for demo detection."""
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "frontend" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def road_scene(name: str, w=960, h=540):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # sky
    img[: h // 2] = (48, 36, 28)
    for y in range(h // 2):
        img[y] = (int(40 + y * 0.08), int(30 + y * 0.05), int(25 + y * 0.03))
    # road
    img[h // 2 :] = (45, 45, 48)
    cv2.rectangle(img, (0, h // 2), (w, h), (42, 42, 46), -1)
    # lane lines
    for x in range(40, w, 80):
        cv2.rectangle(img, (x, h // 2 + 40), (x + 40, h // 2 + 48), (200, 200, 200), -1)
    # sidewalk
    cv2.rectangle(img, (0, h // 2 - 20), (w, h // 2), (70, 70, 75), -1)

    rng = np.random.default_rng(abs(hash(name)) % (2**32))

    def car(x, y, scale=1.0, color=(30, 90, 200)):
        ww, hh = int(120 * scale), int(55 * scale)
        cv2.rectangle(img, (x, y), (x + ww, y + hh), color, -1)
        cv2.rectangle(img, (x + int(15 * scale), y - int(28 * scale)), (x + ww - int(15 * scale), y + 5), (20, 40, 80), -1)
        # windows as "person" silhouettes
        cv2.rectangle(
            img,
            (x + int(25 * scale), y - int(24 * scale)),
            (x + int(50 * scale), y - 2),
            (40, 50, 60),
            -1,
        )
        cv2.circle(img, (x + int(30 * scale), y + hh), int(12 * scale), (20, 20, 20), -1)
        cv2.circle(img, (x + ww - int(30 * scale), y + hh), int(12 * scale), (20, 20, 20), -1)

    def bike(x, y, scale=1.0, with_riders=1):
        # wheels
        cv2.circle(img, (x, y), int(22 * scale), (30, 30, 30), 3)
        cv2.circle(img, (x + int(70 * scale), y), int(22 * scale), (30, 30, 30), 3)
        cv2.line(img, (x, y), (x + int(70 * scale), y), (60, 60, 60), 3)
        # riders
        for i in range(with_riders):
            rx = x + int(15 * scale) + i * int(18 * scale)
            ry = y - int(45 * scale) - i * 2
            cv2.ellipse(img, (rx, ry), (int(12 * scale), int(28 * scale)), 0, 0, 360, (40, 50, 90), -1)
            cv2.circle(img, (rx, ry - int(32 * scale)), int(10 * scale), (60, 70, 100), -1)

    def person(x, y, scale=1.0):
        cv2.circle(img, (x, y - int(40 * scale)), int(12 * scale), (70, 80, 110), -1)
        cv2.rectangle(img, (x - int(12 * scale), y - int(28 * scale)), (x + int(12 * scale), y + int(20 * scale)), (50, 60, 100), -1)

    def traffic_light(x, y):
        cv2.rectangle(img, (x, y), (x + 24, y + 70), (20, 20, 20), -1)
        cv2.circle(img, (x + 12, y + 15), 8, (0, 0, 220), -1)
        cv2.circle(img, (x + 12, y + 35), 8, (0, 180, 180), -1)
        cv2.circle(img, (x + 12, y + 55), 8, (0, 160, 0), -1)

    if "junction" in name:
        traffic_light(w - 80, 80)
        car(120, h // 2 + 60, 1.1, (20, 100, 180))
        car(400, h // 2 + 80, 0.9, (180, 80, 40))
        bike(700, h // 2 + 100, 1.0, with_riders=2)
        person(60, h // 2 + 30, 0.9)
    elif "bike" in name:
        bike(200, h // 2 + 90, 1.3, with_riders=3)
        bike(520, h // 2 + 110, 1.0, with_riders=1)
        car(700, h // 2 + 70, 0.85, (40, 40, 140))
        person(80, h // 2 + 40)
    else:
        car(100, h // 2 + 70, 1.2, (30, 120, 60))
        car(380, h // 2 + 90, 1.0, (160, 50, 50))
        car(650, h // 2 + 60, 0.95, (40, 40, 40))
        person(300, h // 2 + 20)
        bike(800, h // 2 + 100, 0.9, with_riders=2)

    # noise / grain
    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # banner
    cv2.putText(img, f"SafeCityAI sample · {name}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    path = OUT / f"{name}.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print("wrote", path)
    return path


if __name__ == "__main__":
    for n in ("street_cars", "bike_triple", "junction_signal"):
        road_scene(n)
