"""Deterministic stacked degradation: one output per page, every effect applied at once."""

import hashlib
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

VERSION = 8
AUG = "aug"
# 1 draws every knob uniformly across the used range; set above 1 to lean mild again.
BIAS = 1
# Fraction of each documented range that is actually used: 0.5 keeps every knob inside the
# mild half, so nothing reaches the strong end of the table in the research note.
SPAN = 0.5


def augment(image: Image.Image, seed: int) -> tuple[Image.Image, dict]:
    """Apply all effects to one page, each sampled independently from its own range."""
    rng = np.random.default_rng(seed)
    result = image.convert("RGB")
    width, height = result.size
    pixel_scale = min(width, height) / 2480  # A4 short edge at 300 DPI.
    params = {"seed": seed, "version": VERSION, "image_size": [width, height],
              "training_review_required": True, "geometry": {}}

    def knob(mild, strong):
        """One strength per knob, drawn uniformly from `mild` to `mild + SPAN * range`."""
        return float(mild + (strong - mild) * SPAN * rng.uniform(0, 1) ** BIAS)

    def homography(destination, kind, fillcolor):
        """Map the page into `destination` corners; both directions are saved for labels."""
        nonlocal result
        source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
        matrix, target = [], []
        for (x, y), (u, v) in zip(destination, source):
            matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                           [0, 0, 0, x, y, 1, -v * x, -v * y]])
            target.extend([u, v])
        inverse = np.append(np.linalg.solve(matrix, target), 1).reshape(3, 3)
        forward = np.linalg.inv(inverse)
        forward /= forward[2, 2]
        result = result.transform(result.size, Image.Transform.PERSPECTIVE,
                                  inverse.flatten()[:8], Image.Resampling.BICUBIC,
                                  fillcolor=fillcolor)
        params["geometry"][kind] = {"kind": "homography",
                                    "source_to_output": forward.tolist(),
                                    "destination_corners": destination.tolist()}

    # Geometry first, so later photometric effects land on the degraded geometry.
    envelope = knob(0.015, 0.15)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    inset = rng.uniform(0.5, 1, (4, 2)) * [width, height] * envelope
    homography(corners + inset * [[1, 1], [-1, 1], [-1, -1], [1, -1]], "perspective",
               (245, 243, 238))

    skew = knob(0.5, 15) * int(rng.choice([-1, 1]))
    radians = np.deg2rad(skew)
    cosine, sine = np.cos(radians), np.sin(radians)
    scale = min(width / (width * abs(cosine) + height * abs(sine)),
                height / (height * abs(cosine) + width * abs(sine))) * 0.99
    forward = np.eye(3)
    forward[:2, :2] = np.array([[cosine, sine], [-sine, cosine]]) * scale
    center = np.array([width / 2, height / 2])
    forward[:2, 2] = center - forward[:2, :2] @ center
    result = result.transform(result.size, Image.Transform.AFFINE,
                              np.linalg.inv(forward)[:2].flatten(),
                              Image.Resampling.BICUBIC, fillcolor="white")
    params["rotation_degrees"] = skew
    params["geometry"]["skew"] = {"kind": "homography", "source_to_output": forward.tolist()}

    amplitude = knob(0.008, 0.075) * height * int(rng.choice([-1, 1]))
    cycles = float(rng.uniform(1, 2))
    margin_x, margin_y = width * 0.01, abs(amplitude) + height * 0.01
    scale_x, scale_y = 1 - 2 * margin_x / width, 1 - 2 * margin_y / height
    mesh = []
    edges = np.linspace(0, width, 49, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop == start:
            continue
        quad = []
        for u, v in ((start, 0), (start, height), (stop, height), (stop, 0)):
            x = (u - margin_x) / scale_x
            y = (v - margin_y - amplitude * np.sin(np.pi * cycles * x / width)) / scale_y
            quad.extend([float(x), float(y)])
        mesh.append(((int(start), 0, int(stop), height), tuple(quad)))
    result = result.transform(result.size, Image.Transform.MESH, mesh,
                              Image.Resampling.BICUBIC, fillcolor="white")
    params["geometry"]["curvature"] = {"kind": "sine_mesh", "amplitude_pixels": amplitude,
                                       "sine_cycles": cycles,
                                       "margin": [margin_x, margin_y],
                                       "scale": [scale_x, scale_y], "mesh": mesh}

    # shortcut: resolution reduction is disabled — stacked with the rest it destroyed too
    # many pages. Re-enable it at a milder range and bump VERSION again. It also consumed a
    # random draw, so every knob after it lands on a different value for a given seed.
    # resolution = knob(0.80, 0.22)
    # small = (max(1, round(width * resolution)), max(1, round(height * resolution)))
    # result = result.resize(small, Image.Resampling.LANCZOS).resize(
    #     (width, height), Image.Resampling.BICUBIC)
    # params.update(resolution_scale=resolution, intermediate_size=list(small))

    radius = knob(0.4, 2.8) * pixel_scale
    result = result.filter(ImageFilter.GaussianBlur(radius))
    params["gaussian_blur_radius"] = radius

    length = knob(2.5, 14) * pixel_scale
    angle = float(rng.uniform(0, 180))
    dx, dy = length * np.cos(np.deg2rad(angle)), length * np.sin(np.deg2rad(angle))
    offsets = np.linspace(-0.5, 0.5, max(3, int(np.ceil(length)) * 2 + 1))
    summed = np.zeros((height, width, 3), dtype=np.float32)
    for offset in offsets:
        shifted = result.transform(result.size, Image.Transform.AFFINE,
                                   (1, 0, float(dx * offset), 0, 1, float(dy * offset)),
                                   Image.Resampling.BICUBIC, fillcolor="white")
        summed += np.asarray(shifted, dtype=np.float32) / len(offsets)
    result = Image.fromarray(np.clip(summed, 0, 255).astype(np.uint8))
    params.update(motion_blur_pixels=length, motion_angle_degrees=angle,
                  motion_samples=len(offsets))

    mode = str(rng.choice(["thin", "spread"]))
    ink = knob(0.25, 0.88)
    kernel = max(3, 2 * round(int(rng.choice([1, 2])) * pixel_scale) + 1)
    morph = ImageFilter.MaxFilter(kernel) if mode == "thin" else ImageFilter.MinFilter(kernel)
    result = Image.blend(result, result.filter(morph), ink)
    retention = knob(0.95, 0.60)
    result = Image.blend(Image.new("RGB", result.size, "white"), result, retention)
    params.update(ink_mode=mode, ink_strength=ink, ink_kernel=kernel, ink_retention=retention)

    contrast = knob(0.96, 0.60)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    pixels = np.asarray(result, dtype=np.float32).copy()
    shadow = knob(0.14, 0.62)
    horizontal = bool(rng.integers(2))
    gradient = np.linspace(1 - shadow, 1, width if horizontal else height, dtype=np.float32)
    if bool(rng.integers(2)):
        gradient = gradient[::-1]
    pixels *= gradient[None, :, None] if horizontal else gradient[:, None, None]
    moire = knob(4, 28)
    period = max(2.0, float(rng.uniform(4, 9)) * pixel_scale)
    phase = float(rng.uniform(0, 2 * np.pi))
    wave_x = np.sin(np.arange(width, dtype=np.float32) * 2 * np.pi / period + phase)
    wave_y = np.sin(np.arange(height, dtype=np.float32) * 2 * np.pi / (period * 1.13))
    pixels += moire * (wave_x[None, :, None] + wave_y[:, None, None]) / 2
    noise = knob(1.0, 7.0)
    pixels += rng.normal(0, noise, (height, width, 1)).astype(np.float32)
    result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    params.update(contrast=contrast, shadow_strength=shadow, shadow_horizontal=horizontal,
                  shadow_start=float(gradient[0]), shadow_end=float(gradient[-1]),
                  moire_strength=moire, moire_period_pixels=period, moire_phase=phase,
                  noise_std=noise)

    quality = round(knob(86, 20))
    subsampling = int(rng.choice([0, 1, 2]))
    compressed = io.BytesIO()
    result.save(compressed, format="JPEG", quality=quality, subsampling=subsampling)
    with Image.open(io.BytesIO(compressed.getvalue())) as decoded:
        result = decoded.convert("RGB")
    params.update(jpeg_quality=quality, jpeg_subsampling=subsampling)
    return result, params


def page_seed(source_hash: str, page: int, seed: int) -> int:
    digest = hashlib.sha256(f"{source_hash}:{page}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
