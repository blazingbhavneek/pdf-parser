"""Small GPU models behind one HTTP port, each with dynamic batching (stand-in for Triton).

Needs paddlex + paddlepaddle-gpu (the benchmark's Paddle venv has both):
    PYTHONPATH=src ~/.venvs/paddlepaddle_venv/bin/python -m pdf_parser.model_server --port 8119

POST /layout {"image": b64}  -> {"boxes": [{"label", "score", "box": [x0, y0, x1, y1], "order"}]}
POST /det    {"image": b64}  -> {"polys": [[[x, y] * 4]], "scores": [...]}
POST /rec    {"images": [b64], "top_k": 4, "mask": true}
             -> {"lines": [{"text", "chars": [{"c", "p", "alts": [[c, p], ...]}]}]}
Images are PNG/JPEG bytes. Each model is owned by one thread that drains its queue into batches,
so concurrent requests from many pages share GPU batches and memory stays bounded by batch caps.
"""

import argparse
import base64
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time

import cv2
import numpy as np

from pdf_parser.charset import japanese

# PaddleOCR-VL-1.6's layout settings (paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml).
LAYOUT_MERGE = {index: mode for index, mode in enumerate(
    "union union union large union large large union union union union union union union union "
    "large union large union union union union union union union".split())}


class Batcher:
    """One thread owns one model; single requests are collected into batches for it."""

    def __init__(self, name, factory, max_batch, max_wait):
        self.queue = queue.Queue()
        self.ready = threading.Event()
        self.error = None
        threading.Thread(target=self.loop, args=(factory, max_batch, max_wait), name=name, daemon=True).start()

    def submit(self, item):
        future = Future()
        self.queue.put((item, future))
        return future

    def loop(self, factory, max_batch, max_wait):
        try:
            run = factory()  # Built inside the thread that will use it.
        except BaseException as error:
            self.error = error
            self.ready.set()
            raise
        self.ready.set()
        while True:
            batch = [self.queue.get()]
            deadline = time.monotonic() + max_wait
            while len(batch) < max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = run([item for item, _ in batch])
            except Exception as error:
                for _, future in batch:
                    future.set_exception(error)
            else:
                for (_, future), result in zip(batch, results):
                    future.set_result(result)


def decode_ctc(probs, characters, mask, top_k):
    """Greedy CTC over allowed characters, keeping top-k alternatives at each emitted character.

    `p` is the masked probability at the character's most confident frame, so a frame whose
    best class was a disallowed character (say 值) yields a low `p` and becomes a slot later.
    """
    if mask is not None:
        probs = np.where(mask, probs, 0.0)
    best = probs.argmax(-1)
    chars, frame = [], 0
    while frame < len(best):
        index = best[frame]
        start = frame
        while frame < len(best) and best[frame] == index:
            frame += 1
        if index == 0:
            continue
        peak = start + int(probs[start:frame, index].argmax())
        row = probs[peak].copy()
        row[0] = 0.0
        total = float(row.sum()) or 1.0
        ranked = np.argsort(-row)[:top_k]
        chars.append({"c": characters[index], "p": float(probs[peak, index]),
                      "alts": [[characters[k], float(row[k]) / total] for k in ranked if row[k] > 0]})
    return {"text": "".join(c["c"] for c in chars), "chars": chars}


def layout_factory(model_name, device, threshold):
    def factory():
        from paddlex import create_model

        model = create_model(model_name, device=device)

        def run(images):
            results = model.predict(images, batch_size=len(images), threshold=threshold, layout_nms=True,
                                    layout_unclip_ratio=[1.0, 1.0], layout_merge_bboxes_mode=LAYOUT_MERGE)
            # Boxes come sorted by the model's predicted reading order; PaddleX blanks "order" for
            # tables, captions, headers, ... only for display, so the position is the order of every box.
            return [{"boxes": [{"label": box["label"], "score": float(box["score"]),
                                "box": [float(v) for v in box["coordinate"]], "order": rank}
                               for rank, box in enumerate(result["boxes"])]} for result in results]
        return run
    return factory


def det_factory(model_name, device):
    def factory():
        from paddlex import create_model

        model = create_model(model_name, device=device)

        def run(images):
            # Detection keeps the aspect ratio, so only pages of one size can share a batch.
            results = [None] * len(images)
            groups = {}
            for k, image in enumerate(images):
                groups.setdefault(image.shape, []).append(k)
            for indexes in groups.values():
                for k, result in zip(indexes, model.predict([images[k] for k in indexes], batch_size=len(indexes))):
                    results[k] = {"polys": np.asarray(result["dt_polys"], dtype=float).reshape(-1, 4, 2).tolist(),
                                  "scores": [float(s) for s in result["dt_scores"]]}
            return results
        return run
    return factory


def rec_factory(model_name, device, chunk):
    def factory():
        from paddlex import create_model

        model = create_model(model_name, device=device)
        characters = list(model.post_op.character)  # index 0 is the CTC blank
        allowed = japanese()
        mask = np.array([index == 0 or c in allowed for index, c in enumerate(characters)])

        def run(items):
            images = []
            for image, _, _ in items:
                height, width = image.shape[:2]
                images.append(np.rot90(image) if height >= 1.5 * width else image)  # vertical line
            results = [None] * len(images)
            # Similar aspect ratios share a batch, so padding stays small.
            order = sorted(range(len(images)), key=lambda k: images[k].shape[1] / images[k].shape[0])
            for start in range(0, len(order), chunk):
                indexes = order[start:start + chunk]
                batch = model.pre_tfs["ToBatch"](imgs=model.pre_tfs["ReisizeNorm"](imgs=[images[k] for k in indexes]))
                probs = model.runner(x=batch)[0]
                for row, k in zip(probs, indexes):
                    _, top_k, use_mask = items[k]
                    results[k] = decode_ctc(row, characters, mask if use_mask else None, top_k)
            return results
        return run
    return factory


def image(data):
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Undecodable image")
    return decoded  # BGR, as PaddleX expects for arrays


class Handler(BaseHTTPRequestHandler):
    models = {}

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {"ok": True}) if self.path == "/health" else self.reply(404, {"error": "not found"})

    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if self.path in ("/layout", "/det"):
                self.reply(200, self.models[self.path].submit(image(request["image"])).result())
            elif self.path == "/rec":
                top_k, mask = int(request.get("top_k", 4)), bool(request.get("mask", True))
                futures = [self.models["/rec"].submit((image(data), top_k, mask)) for data in request["images"]]
                self.reply(200, {"lines": [future.result() for future in futures]})
            else:
                self.reply(404, {"error": "not found"})
        except Exception as error:
            self.reply(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, format, *args):
        pass  # One line per request drowns the console at hundreds of requests per second.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8119)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--layout-model", default="PP-DocLayoutV3")
    parser.add_argument("--det-model", default="PP-OCRv6_medium_det")
    parser.add_argument("--rec-model", default="PP-OCRv6_medium_rec")
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--layout-batch", type=int, default=8)
    parser.add_argument("--det-batch", type=int, default=8)
    parser.add_argument("--rec-batch", type=int, default=256, help="line crops per GPU pass group")
    parser.add_argument("--rec-chunk", type=int, default=32, help="crops per forward pass")
    parser.add_argument("--max-wait-ms", type=float, default=5.0)
    args = parser.parse_args()
    wait = args.max_wait_ms / 1000
    Handler.models = {
        "/layout": Batcher("layout", layout_factory(args.layout_model, args.device, args.layout_threshold),
                           args.layout_batch, wait),
        "/det": Batcher("det", det_factory(args.det_model, args.device), args.det_batch, wait),
        "/rec": Batcher("rec", rec_factory(args.rec_model, args.device, args.rec_chunk), args.rec_batch, wait),
    }
    for name, batcher in Handler.models.items():
        batcher.ready.wait()
        if batcher.error:
            raise SystemExit(f"{name} failed to load: {batcher.error}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Model server ready on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
