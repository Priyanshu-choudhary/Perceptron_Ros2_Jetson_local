"""Tests for the Stage 8 costmap scoring helpers (costmap_eval.py)."""

import math

import numpy as np

from perceptron_perception import costmap_eval as ke


def test_cells_in_rotated_box():
    ii, jj = ke.cells_in_box(np.array([0.0, 0.0]), 0.2, (100, 100), (10.0, 10.0), math.pi / 2, (4.0, 2.0))
    xs = (ii + 0.5) * 0.2
    ys = (jj + 0.5) * 0.2
    assert xs.max() - xs.min() < 2.4 and ys.max() - ys.min() > 3.6       # long side now along y


def test_flicker_aligns_shifted_grids():
    a = np.zeros((10, 10), np.int8)
    a[5, 5] = 100
    b = np.zeros((10, 10), np.int8)
    b[4, 5] = 100                       # same world cell; the window moved one cell in +x
    assert ke.flicker((a, np.array([0.0, 0.0]), 1.0), (b, np.array([1.0, 0.0]), 1.0)) == 0.0
    c = np.zeros((10, 10), np.int8)     # it disappeared
    assert ke.flicker((a, np.array([0.0, 0.0]), 1.0), (c, np.array([0.0, 0.0]), 1.0)) > 0.0


def test_value_at_bounds():
    g = np.arange(9, dtype=np.int8).reshape(3, 3)
    assert ke.value_at(g, np.array([0.0, 0.0]), 1.0, (2.5, 0.5)) == 6
    assert ke.value_at(g, np.array([0.0, 0.0]), 1.0, (5.0, 0.5)) is None
