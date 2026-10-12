"""Deterministic scan/camera degradation; keep the entire page in view."""

import hashlib
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

VERSION = 2
LEGACY_PROFILES = ("scan", "motion", "photo")
PROFILES = (*LEGACY_PROFILES, "resolution", "compression", "ink", "warp", "screen")
SEVERITIES = ("mild", "medium", "hard")


def augment(image: Image.Image, profile: str, seed: int,
            severity: str | None = None) -> tuple[Image.Image, dict]:
    # Existing manifests must repair to the same pixels, never replace benchmark inputs.
    if severity is None:
        return _legacy_augment(image, profile, seed)
    if profile not in PROFILES or severity not in SEVERITIES:
        raise ValueError(f"Unknown augmentation/severity: {profile}/{severity}")
    rng = np.random.default_rng(seed)
    level = SEVERITIES.index(severity)
    result = image.convert("RGB")
    width, height = result.size
    pixel_scale = min(width, height) / 2480  # A4 short edge at 300 DPI.
    params = {"profile": profile, "severity": severity, "seed": seed, "version": VERSION,
              "image_size": [width, height], "training_review_required": True,
              "geometry": {"kind": "identity"}}

    def sample(mild, medium, hard):
        return float(rng.uniform(*(mild, medium, hard)[level]))

    if profile in ("photo", "screen"):
        fraction = sample((0.01, 0.025), (0.03, 0.06), (0.07, 0.11))
        inset = rng.uniform(0.5, 1, (4, 2)) * [width, height] * fraction
        source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
        destination = source + inset * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
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
                                  fillcolor=(245, 243, 238))
        params["geometry"] = {"kind": "homography", "source_to_output": forward.tolist(),
                              "destination_corners": destination.tolist()}
    elif profile == "scan":
        angle = sample((0.2, 0.7), (1, 2), (2.5, 4)) * int(rng.choice([-1, 1]))
        radians = np.deg2rad(angle)
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
        params["rotation_degrees"] = angle
        params["geometry"] = {"kind": "homography", "source_to_output": forward.tolist()}
    elif profile == "warp":
        amplitude = sample((0.005, 0.012), (0.018, 0.03), (0.04, 0.065)) * height
        amplitude *= int(rng.choice([-1, 1]))
        margin_x, margin_y = width * 0.01, abs(amplitude) + height * 0.01
        scale_x, scale_y = 1 - 2 * margin_x / width, 1 - 2 * margin_y / height
        mesh = []
        for start, stop in zip(np.linspace(0, width, 49, dtype=int)[:-1],
                               np.linspace(0, width, 49, dtype=int)[1:]):
            if stop == start:
                continue
            quad = []
            for u, v in ((start, 0), (start, height), (stop, height), (stop, 0)):
                x = (u - margin_x) / scale_x
                y = (v - margin_y - amplitude * np.sin(np.pi * x / width)) / scale_y
                quad.extend([float(x), float(y)])
            mesh.append(((int(start), 0, int(stop), height), tuple(quad)))
        result = result.transform(result.size, Image.Transform.MESH, mesh,
                                  Image.Resampling.BICUBIC, fillcolor="white")
        params["geometry"] = {"kind": "sine_mesh", "amplitude_pixels": amplitude,
                              "margin": [margin_x, margin_y], "scale": [scale_x, scale_y],
                              "mesh": mesh}

    if profile in ("scan", "photo", "screen", "resolution"):
        scale = sample((0.72, 0.88), (0.45, 0.62), (0.25, 0.38))
        small = (max(1, round(width * scale)), max(1, round(height * scale)))
        result = result.resize(small, Image.Resampling.LANCZOS).resize(
            (width, height), Image.Resampling.BICUBIC)
        params.update(resolution_scale=scale, intermediate_size=list(small))

    if profile in ("scan", "photo"):
        radius = sample((0.25, 0.5), (0.7, 1.1), (1.5, 2.2)) * pixel_scale
        result = result.filter(ImageFilter.GaussianBlur(radius))
        params["gaussian_blur_radius"] = radius
    elif profile == "motion":
        length = sample((1.5, 2.5), (4, 6), (8, 12)) * pixel_scale
        angle = float(rng.uniform(0, 180))
        dx, dy = length * np.cos(np.deg2rad(angle)), length * np.sin(np.deg2rad(angle))
        offsets = np.linspace(-0.5, 0.5, max(3, int(np.ceil(length)) * 2 + 1))
        pixels = np.zeros((height, width, 3), dtype=np.float32)
        for offset in offsets:
            shifted = result.transform(result.size, Image.Transform.AFFINE,
                                       (1, 0, float(dx * offset), 0, 1, float(dy * offset)),
                                       Image.Resampling.BICUBIC, fillcolor="white")
            pixels += np.asarray(shifted, dtype=np.float32) / len(offsets)
        result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        params.update(motion_blur_pixels=length, motion_angle_degrees=angle,
                      motion_samples=len(offsets))
    elif profile == "ink":
        mode = str(rng.choice(["thin", "spread"]))
        strength = sample((0.15, 0.3), (0.4, 0.55), (0.65, 0.8))
        kernel = max(3, 2 * round(pixel_scale) + 1)
        morph = ImageFilter.MaxFilter(kernel) if mode == "thin" else ImageFilter.MinFilter(kernel)
        result = Image.blend(result, result.filter(morph), strength)
        fade = sample((0.94, 0.98), (0.8, 0.9), (0.65, 0.75))
        result = Image.blend(Image.new("RGB", result.size, "white"), result, fade)
        params.update(ink_mode=mode, ink_strength=strength, ink_kernel=kernel, ink_retention=fade)

    if profile in ("scan", "photo", "screen"):
        contrast = sample((0.94, 0.99), (0.82, 0.92), (0.65, 0.78))
        result = ImageEnhance.Contrast(result).enhance(contrast)
        pixels = np.asarray(result, dtype=np.float32).copy()
        if profile == "photo":
            strength = sample((0.08, 0.16), (0.22, 0.35), (0.4, 0.55))
            horizontal = bool(rng.integers(2))
            gradient = np.linspace(1 - strength, 1, width if horizontal else height,
                                   dtype=np.float32)
            if bool(rng.integers(2)):
                gradient = gradient[::-1]
            pixels *= gradient[None, :, None] if horizontal else gradient[:, None, None]
            params.update(shadow_strength=strength, shadow_horizontal=horizontal,
                          shadow_start=float(gradient[0]), shadow_end=float(gradient[-1]))
        elif profile == "screen":
            strength = sample((2, 5), (7, 12), (16, 24))
            period = max(2.0, float(rng.uniform(4, 9)) * pixel_scale)
            phase = float(rng.uniform(0, 2 * np.pi))
            horizontal = np.sin(np.arange(width, dtype=np.float32) * 2 * np.pi / period + phase)
            vertical = np.sin(np.arange(height, dtype=np.float32) * 2 * np.pi / (period * 1.13))
            pixels += strength * (horizontal[None, :, None] + vertical[:, None, None]) / 2
            params.update(moire_strength=strength, moire_period_pixels=period, moire_phase=phase)
        noise = sample((0.5, 1.2), (2, 3.5), (4, 6))
        pixels += rng.normal(0, noise, (height, width, 1)).astype(np.float32)
        result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        params.update(contrast=contrast, noise_std=noise)

    if profile in ("scan", "photo", "screen", "compression"):
        quality = round(sample((78, 90), (48, 65), (22, 38)))
        compressed = io.BytesIO()
        result.save(compressed, format="JPEG", quality=quality, subsampling=2)
        with Image.open(io.BytesIO(compressed.getvalue())) as decoded:
            result = decoded.convert("RGB")
        params.update(jpeg_quality=quality, jpeg_subsampling=2)
    return result, params


def _legacy_augment(image: Image.Image, profile: str, seed: int) -> tuple[Image.Image, dict]:
    if profile not in LEGACY_PROFILES:
        raise ValueError(f"Unknown augmentation: {profile}")
    rng = np.random.default_rng(seed)
    result = image.convert("RGB")
    width, height = result.size
    params = {"profile": profile, "seed": seed, "version": 1}

    if profile == "photo":
        # Map the complete source rectangle into an inset quadrilateral: no cropped labels.
        inset = rng.uniform(0.015, 0.055, (4, 2)) * [width, height]
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
        destination = corners + inset * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        matrix, target = [], []
        for (x, y), (u, v) in zip(destination, corners):
            matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                           [0, 0, 0, x, y, 1, -v * x, -v * y]])
            target.extend([u, v])
        coefficients = np.linalg.solve(matrix, target)
        result = result.transform(result.size, Image.Transform.PERSPECTIVE, coefficients,
                                  Image.Resampling.BICUBIC, fillcolor=(238, 236, 230))
        params["corners"] = destination.tolist()
    else:
        angle = float(rng.uniform(-1.2, 1.2))
        rotated = result.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        rotated.thumbnail(result.size, Image.Resampling.LANCZOS)
        result = Image.new("RGB", result.size, "white")
        result.paste(rotated, ((width - rotated.width) // 2, (height - rotated.height) // 2))
        params["rotation_degrees"] = angle

    scale = float(rng.uniform(0.62, 0.85))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    result = result.resize(small_size, Image.Resampling.LANCZOS).resize(
        (width, height), Image.Resampling.BICUBIC)
    params["resolution_scale"] = scale

    if profile == "motion":
        kernel = [0.0] * 25
        kernel[10:15] = [0.2] * 5
        result = result.filter(ImageFilter.Kernel((5, 5), kernel, scale=1))
        params["motion_blur_pixels"] = 5
    else:
        radius = float(rng.uniform(0.35, 0.8))
        result = result.filter(ImageFilter.GaussianBlur(radius))
        params["gaussian_blur_radius"] = radius

    contrast = float(rng.uniform(0.78, 0.95))
    result = ImageEnhance.Contrast(result).enhance(contrast)
    noise_std = float(rng.uniform(1.0, 3.5))
    pixels = np.asarray(result, dtype=np.float32)
    if profile == "photo":
        shadow_strength = float(rng.uniform(0.12, 0.28))
        gradient = np.linspace(1 - shadow_strength, 1, width, dtype=np.float32)
        pixels *= gradient[None, :, None]
        params["shadow_strength"] = shadow_strength
    pixels += rng.normal(0, noise_std, (height, width, 1)).astype(np.float32)
    result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    quality = int(rng.integers(60, 83))
    compressed = io.BytesIO()
    result.save(compressed, format="JPEG", quality=quality)
    params.update(contrast=contrast, noise_std=noise_std, jpeg_quality=quality)
    with Image.open(io.BytesIO(compressed.getvalue())) as decoded:
        return decoded.copy(), params


def page_seed(source_hash: str, page: int, profile: str, seed: int) -> int:
    digest = hashlib.sha256(f"{source_hash}:{page}:{profile}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
