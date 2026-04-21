"""5-line quickstart: one image in, 34 trait values + diagnostic figure out.

Run:
    pip install face-trait-transformer[hub,figures]
    python examples/quickstart.py path/to/your/face.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

from face_trait_transformer import TraitPredictor


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python quickstart.py path/to/face.jpg")
        return 1
    img = Path(sys.argv[1])
    predictor = TraitPredictor.from_pretrained()  # defaults to kiante/face-trait-transformer
    row, fig = predictor.predict_with_figure(img, out_path="diag.png")
    print(row.to_string())
    print("wrote diag.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
