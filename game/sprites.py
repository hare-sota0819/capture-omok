"""Anti-aliased images for the GUI, drawn pixel by pixel and encoded as PNG.

Tkinter's canvas draws circles with jagged edges and has no transparency, so
the stones, rounded panels and the wooden board are rendered here instead,
using only the standard library. Every function returns (width, height, RGBA
bytes); encode_png() turns that into data tk.PhotoImage can load.
"""

import math
import random
import struct
import zlib


def encode_png(width, height, rgba):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    stride = width * 4
    rows = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6))
            + chunk(b"IEND", b""))


def _smoothstep(edge0, edge1, x):
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t = (x - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)


def with_alpha(image, factor):
    """The same image, more transparent."""
    width, height, rgba = image
    faded = bytearray(rgba)
    faded[3::4] = bytes(int(a * factor) for a in rgba[3::4])
    return width, height, bytes(faded)


def stone(radius, light, dark, samples=3):
    """A shaded stone lit from the upper left, with a soft drop shadow."""
    pad = int(math.ceil(radius * 0.3)) + 2
    size = 2 * int(math.ceil(radius)) + 2 * pad
    centre = size / 2.0
    light_x, light_y = centre - 0.38 * radius, centre - 0.42 * radius
    shadow_x, shadow_y = centre + 0.05 * radius, centre + 0.15 * radius
    step = 1.0 / samples
    count = samples * samples
    pixels = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            red = green = blue = alpha = 0.0
            for j in range(samples):
                py = y + (j + 0.5) * step
                for i in range(samples):
                    px = x + (i + 0.5) * step
                    distance = math.hypot(px - centre, py - centre)
                    if distance <= radius:
                        t = min(1.0, math.hypot(px - light_x, py - light_y) / (1.55 * radius))
                        t = t ** 0.85
                        rim = 1.0 - 0.22 * _smoothstep(0.82 * radius, radius, distance)
                        red += (light[0] + (dark[0] - light[0]) * t) * rim
                        green += (light[1] + (dark[1] - light[1]) * t) * rim
                        blue += (light[2] + (dark[2] - light[2]) * t) * rim
                        alpha += 1.0
                    else:
                        spread = math.hypot(px - shadow_x, py - shadow_y)
                        alpha += 0.42 * (1.0 - _smoothstep(0.8 * radius, 1.22 * radius, spread))
            if alpha > 0.0:
                index = (y * size + x) * 4
                pixels[index] = min(255, int(red / alpha + 0.5))
                pixels[index + 1] = min(255, int(green / alpha + 0.5))
                pixels[index + 2] = min(255, int(blue / alpha + 0.5))
                pixels[index + 3] = int(alpha / count * 255.0 + 0.5)
    return size, size, bytes(pixels)


def _radial(extent, rgb, alpha_at, samples=3):
    """A single-colour image whose opacity depends only on distance from the centre."""
    size = 2 * int(math.ceil(extent)) + 2
    centre = size / 2.0
    step = 1.0 / samples
    count = samples * samples
    pixels = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            alpha = 0.0
            for j in range(samples):
                py = y + (j + 0.5) * step
                for i in range(samples):
                    alpha += alpha_at(math.hypot(x + (i + 0.5) * step - centre, py - centre))
            if alpha > 0.0:
                index = (y * size + x) * 4
                pixels[index:index + 3] = bytes(rgb)
                pixels[index + 3] = int(alpha / count * 255.0 + 0.5)
    return size, size, bytes(pixels)


def disc(radius, rgb):
    return _radial(radius, rgb, lambda d: 1.0 if d <= radius else 0.0)


def ring(radius, thickness, rgb, glow=0.0):
    """A circle outline, optionally with a soft glow outside it."""
    inner, outer = radius - thickness / 2.0, radius + thickness / 2.0

    def alpha_at(distance):
        if inner <= distance <= outer:
            return 1.0
        if glow and distance > outer:
            return 0.55 * (1.0 - _smoothstep(outer, outer + glow, distance))
        return 0.0

    return _radial(outer + glow, rgb, alpha_at)


def _corner_coverage(radius, samples=4):
    """coverage[y][x]: how much of pixel (x, y) of a top-left corner is inside."""
    step = 1.0 / samples
    count = float(samples * samples)
    table = []
    for y in range(radius):
        row = []
        for x in range(radius):
            inside = 0
            for j in range(samples):
                for i in range(samples):
                    if math.hypot(x + (i + 0.5) * step - radius,
                                  y + (j + 0.5) * step - radius) <= radius:
                        inside += 1
            row.append(inside / count)
        table.append(row)
    return table


def _round_corners(pixels, width, height, radius):
    """Make the four corners of an RGBA bytearray transparent, with smooth edges."""
    radius = max(0, min(radius, width // 2, height // 2))
    coverage = _corner_coverage(radius)
    for y in range(radius):
        for x in range(radius):
            amount = coverage[y][x]
            if amount >= 1.0:
                continue
            for px, py in ((x, y), (width - 1 - x, y), (x, height - 1 - y),
                           (width - 1 - x, height - 1 - y)):
                index = (py * width + px) * 4 + 3
                pixels[index] = int(pixels[index] * amount + 0.5)


def rounded_rect(width, height, radius, rgba):
    pixels = bytearray(bytes(rgba) * (width * height))
    _round_corners(pixels, width, height, radius)
    return width, height, bytes(pixels)


def _value_noise(columns, rows, generator):
    """Smooth random values: a random lattice with interpolation between points."""
    lattice = [[generator.random() for _ in range(columns + 2)] for _ in range(rows + 2)]

    def sample(x, y):
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        top = lattice[y0][x0] + (lattice[y0][x0 + 1] - lattice[y0][x0]) * fx
        bottom = lattice[y0 + 1][x0] + (lattice[y0 + 1][x0 + 1] - lattice[y0 + 1][x0]) * fx
        return top + (bottom - top) * fy

    return sample


def wood(width, height, radius, base=(224, 184, 116), seed=7):
    """A wooden board: fine grain running left to right plus slow tonal drift."""
    generator = random.Random(seed)
    block = 4                         # the grain changes every `block` pixels along x
    grain_x, grain_y = 110.0, 2.6     # long along x, tight along y: streaks
    drift_x, drift_y = 260.0, 70.0
    grain = _value_noise(int(width / grain_x) + 1, int(height / grain_y) + 1, generator)
    drift = _value_noise(int(width / drift_x) + 1, int(height / drift_y) + 1, generator)
    blocks = range(0, width, block)
    pixels = bytearray()
    for y in range(height):
        row = bytearray()
        for x in blocks:
            shade = (grain(x / grain_x, y / grain_y) - 0.5) * 20.0 \
                + (drift(x / drift_x, y / drift_y) - 0.5) * 26.0
            pixel = bytes((
                max(0, min(255, int(base[0] + shade))),
                max(0, min(255, int(base[1] + shade * 0.95))),
                max(0, min(255, int(base[2] + shade * 0.8))),
                255,
            ))
            row += pixel * block
        pixels += row[:width * 4]
    _round_corners(pixels, width, height, radius)
    return width, height, bytes(pixels)
