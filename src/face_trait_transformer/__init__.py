"""face-trait-transformer — predict 34-d perceived-trait vectors from face images.

Basic usage:

    >>> from face_trait_transformer import TraitPredictor
    >>> predictor = TraitPredictor.from_pretrained("kiante/face-trait-transformer")
    >>> row = predictor.predict("face.jpg")            # pandas.Series, 34 attributes, 0-100
    >>> df  = predictor.predict(["a.jpg", "b.jpg"])    # batch

Offline bundle (no internet after first download):

    >>> predictor = TraitPredictor.from_bundle("path/to/bundle")

See docs/quickstart.md for more.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .predictor import TraitPredictor

__all__ = ["TraitPredictor", "__version__"]
