"""Run with PADDLE_VENV/bin/python, never the benchmark environment."""

import sys

from paddleocr import PPStructureV3


if __name__ == "__main__":
    pipeline = PPStructureV3(lang="japan")
    for result in pipeline.predict(input=sys.argv[1]):
        result.save_to_json(save_path=sys.argv[2])
        result.save_to_markdown(save_path=sys.argv[2])
