import argparse
import sys
import math
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image


class Pixel:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


def read_img(filename: Path) -> Tuple[List[List["Pixel"]], int, int]:
    """Open ANY image (L, P, RGBA, …) and return Pixel grid plus w & h."""
    img = Image.open(filename).convert("RGB")          #  fix if not rgb image
    w, h = img.size
    flat = list(img.getdata())                         # [(r,g,b), …]
    pixels = [[Pixel(*flat[i * w + j]) for j in range(w)] for i in range(h)]
    return pixels, w, h


def write_img(image, w, h, filename):
    pixels = [tuple([pixel.r, pixel.g, pixel.b]) for row in image for pixel in row]
    img = Image.new("RGB", (w, h))
    img.putdata(pixels)
    img.save(filename, quality=100)


def compute_displacement_field(amp, sigma, w, h):
    d_x = [[0.0 for _ in range(w)] for _ in range(h)]
    d_y = [[0.0 for _ in range(w)] for _ in range(h)]

    da_x = [[0.0 for _ in range(w)] for _ in range(h)]
    da_y = [[0.0 for _ in range(w)] for _ in range(h)]

    kws = int(2.0 * sigma)
    ker = [math.exp(-float(k * k) / (sigma * sigma)) for k in range(-kws, kws + 1)]

    for i in range(h):
        for j in range(w):
            d_x[i][j] = -1.0 + 2.0 * random.random()
            d_y[i][j] = -1.0 + 2.0 * random.random()

    for i in range(h):
        for j in range(w):
            sum_x = 0.0
            sum_y = 0.0
            for k in range(-kws, kws + 1):
                v = j + k
                if v < 0:
                    v = -v
                if v >= w:
                    v = 2 * w - v - 1
                sum_x += d_x[i][v] * ker[abs(k)]
                sum_y += d_y[i][v] * ker[abs(k)]
            da_x[i][j] = sum_x
            da_y[i][j] = sum_y

    for j in range(w):
        for i in range(h):
            sum_x = 0.0
            sum_y = 0.0
            for k in range(-kws, kws + 1):
                u = i + k
                if u < 0:
                    u = -u
                if u >= h:
                    u = 2 * h - u - 1
                sum_x += da_x[u][j] * ker[abs(k)]
                sum_y += da_y[u][j] * ker[abs(k)]
            d_x[i][j] = sum_x
            d_y[i][j] = sum_y

    avg = sum(math.sqrt(d_x[i][j] ** 2 + d_y[i][j] ** 2) for i in range(h) for j in range(w)) / (h * w)

    for i in range(h):
        for j in range(w):
            d_x[i][j] = amp * d_x[i][j] / avg
            d_y[i][j] = amp * d_y[i][j] / avg

    return d_x, d_y


def apply_displacement_field(input, d_x, d_y, w, h):
    output = [[Pixel(0, 0, 0) for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            p1 = i + d_y[i][j]
            p2 = j + d_x[i][j]

            u0 = int(math.floor(p1))
            v0 = int(math.floor(p2))

            f1 = p1 - u0
            f2 = p2 - v0

            sumr, sumg, sumb = 0.0, 0.0, 0.0
            for idx in range(4):
                if idx == 0:
                    u, v = u0, v0
                    f = (1.0 - f1) * (1.0 - f2)
                elif idx == 1:
                    u, v = u0 + 1, v0
                    f = f1 * (1.0 - f2)
                elif idx == 2:
                    u, v = u0, v0 + 1
                    f = (1.0 - f1) * f2
                else:
                    u, v = u0 + 1, v0 + 1
                    f = f1 * f2

                u = max(0, min(u, h - 1))
                v = max(0, min(v, w - 1))

                val = input[u][v].r
                sumr += f * val

                val = input[u][v].g
                sumg += f * val

                val = input[u][v].b
                sumb += f * val

            output[i][j].r = int(sumr)
            output[i][j].g = int(sumg)
            output[i][j].b = int(sumb)

    return output


def rubbersheet(input, w, h, amp, sigma):
    if sigma > h / 2.5 or sigma > w / 2.5:
        sys.stderr.write("- Warning: Gaussian smoothing kernel too large for the input image.\n")
        return

    if sigma < 1E-5:
        sys.stderr.write("- Warning: Gaussian smoothing kernel with negative/zero spread.\n")
        return

    d_x, d_y = compute_displacement_field(amp, sigma, w, h)
    output = apply_displacement_field(input, d_x, d_y, w, h)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Image file OR *root folder* containing images")
    parser.add_argument("amp", type=float)
    parser.add_argument("sigma", type=float)
    parser.add_argument("--outdir", default=".", help="Where to write distorted images")
    parser.add_argument("--ext",
                        default=".jpg,.jpeg,.png,.bmp",
                        help="Comma‑separated extensions of images to process")
    args = parser.parse_args()

    src    = Path(args.src)
    outdir = Path(args.outdir)
    exts   = tuple(e.lower() for e in args.ext.split(","))

    if src.is_dir():
        files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        if not files:
            sys.exit(f"No images with extensions {exts} found under {src}")
    elif src.is_file():
        files = [src]
    else:
        sys.exit(f"{src} is neither a file nor a directory")

    for f in files:
        process_one(f, src, args.amp, args.sigma, outdir)


def process_one(path: Path, root: Path, amp: float, sigma: float, outdir: Path) -> None:
    img, w, h = read_img(path)
    distorted = rubbersheet(img, w, h, amp, sigma)

    rel   = path.relative_to(root)              # e.g. alef/char_12.jpg
    dst   = outdir / rel.parent                 # augmented_data/alef/
    dst.mkdir(parents=True, exist_ok=True)

    outfile = dst / f"{path.stem}_morphed{path.suffix}"
    write_img(distorted, w, h, outfile)
    print(f"✔ {rel}  →  {outfile}")


if __name__ == "__main__":
    main()