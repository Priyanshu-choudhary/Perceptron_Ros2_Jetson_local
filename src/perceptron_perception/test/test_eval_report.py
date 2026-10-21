"""Tests for the evaluators' JSON summaries (eval_report.py)."""

import json

import numpy as np

from perceptron_perception import eval_report as er


def test_jsonable_handles_tuple_keys_numpy_and_nan(tmp_path):
    data = {('vehicle', True): (np.int64(3), np.float32(97.5)), 1.0: [float('nan'), np.array([1, 2])],
            'nested': {('pedestrian', 'moving'): [{'h': 2.0, 'in95': np.float64(np.inf)}]}}
    path = tmp_path / 'sub' / 'x.json'
    er.write_json(str(path), data)
    back = json.loads(path.read_text())
    assert back['vehicle True'] == [3, 97.5]
    assert back['1.0'] == [None, [1, 2]]
    assert back['nested']['pedestrian moving'][0]['in95'] is None
    assert not (tmp_path / 'sub' / 'x.json.tmp').exists()
