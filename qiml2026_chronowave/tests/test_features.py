import numpy as np
import torch

from chronowave.features import haar_approximation_level2, sinusoidal_time_encoding


def test_level2_haar_known_vector():
    x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    y = haar_approximation_level2(x)
    assert y.shape == (1, 1)
    assert np.allclose(y[0, 0], 2.0, atol=1e-6)


def test_level2_haar_odd_length_is_finite():
    x = np.arange(15, dtype=np.float32).reshape(1, -1)
    y = haar_approximation_level2(x)
    assert y.shape[1] == 4
    assert np.isfinite(y).all()


def test_time_encoding_shape_and_t0():
    t = torch.tensor([0.0, 1.0])
    e = sinusoidal_time_encoding(t, 8)
    assert e.shape == (2, 8)
    assert torch.allclose(e[0, 0::2], torch.zeros(4))
    assert torch.allclose(e[0, 1::2], torch.ones(4))
