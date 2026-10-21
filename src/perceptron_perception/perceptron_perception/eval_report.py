"""Evaluator results as JSON files, for the Stage 9 scenario suite (no ROS).

Every *_eval_node writes its since-start summary to its `summary_file` parameter at each
report and at shutdown, so a run that is killed still leaves its last result.
"""

import json
import math
import os

import numpy as np


def jsonable(x):
    """Summaries -> JSON-safe values: tuple keys joined with spaces, numpy scalars and
    arrays to Python, NaN / inf to None."""
    if isinstance(x, dict):
        return {(' '.join(str(p) for p in k) if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def write_json(path, data):
    """Atomic: a reader never sees half a file."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(jsonable(data), f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:                      # Ctrl-C mid-write: keep the last good file, drop the partial one
        if os.path.exists(tmp):
            os.remove(tmp)
