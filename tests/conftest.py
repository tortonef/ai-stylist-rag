import re
import zlib

import numpy as np
import pytest

from ai_stylist.catalog import load_catalog
from ai_stylist.config import load_config
from ai_stylist.query import load_vocabulary


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder: no model download in unit tests."""

    calls = 0

    def encode(self, texts):
        FakeEncoder.calls += 1
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def products(cfg):
    return load_catalog(cfg["paths"]["catalog"], cfg["paths"]["schema"])


@pytest.fixture(scope="session")
def vocab(cfg):
    return load_vocabulary(cfg["paths"]["vocabulary"])


@pytest.fixture(scope="session")
def encoder():
    return FakeEncoder()


@pytest.fixture(scope="session")
def vectors(products, encoder):
    from ai_stylist.catalog import product_text

    return encoder.encode([product_text(p) for p in products])
