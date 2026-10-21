"""Scoring predictions against where CARLA actors really went (roadmap Stage 7), no ROS.

Each prediction made at time t is matched to a true actor by position at t (as in
Stage 6), then held until the ground truth for t + h arrives. For every horizon h:

  error      |predicted centre - true centre at t + h|, ground plane
  coverage   share of true positions inside the predicted 95 % ellipse
             (Mahalanobis^2 <= 5.99): a calibrated predictor scores ~95 %. Much lower
             means the ellipses are too small for Stage 8 to rely on; much higher
             means they are needlessly large and will block free space.
  baselines  computed from the SAME track state at t:
             stay     the object stays where it is
             raw cv   position + raw tracker velocity * h, no gate/shrinkage/caps
             so the table shows what the constraints buy over the naive choices.

Rows are split by class and by whether the actor was really moving at t (true
speed >= 0.2 m/s), because a predictor that is good on parked cars and bad on
walkers would otherwise average out to "fine".
"""

import numpy as np

from perceptron_perception.tracker import class_group

CHI2_95_2D = 5.991


class PredictionScores:

    def __init__(self, horizons):
        self.horizons = list(horizons)
        self.pending = []              # (target_t, key, actor, pred, stay, raw, cov, model)
        self.rows = {}
        self.models = {}
        self.by_model = {}             # (key, model) -> [(ours_err, raw_err)] at the last horizon

    def add_predictions(self, t, items):
        """items: [dict(actor, cls, moving, model, means (H,2), covs (H,2,2), pos (2,), vel (2,))]."""
        for it in items:
            key = (class_group(it['cls']), 'moving' if it['moving'] else 'static')
            self.models[(key, it['model'])] = self.models.get((key, it['model']), 0) + 1
            for k, h in enumerate(self.horizons):
                self.pending.append((round(t + h, 3), key, it['actor'], k, it['means'][k],
                                     it['pos'], it['pos'] + it['vel'] * h, it['covs'][k], it['model']))

    def add_truth(self, t, positions):
        """positions: actor -> xy at time t. Scores every prediction that targeted t."""
        t = round(t, 3)
        keep = []
        for item in self.pending:
            target, key, actor, k, pred, stay, raw, cov, model = item
            if target > t + 1e-6:
                keep.append(item)
                continue
            if target < t - 1e-6 or actor not in positions:
                continue                         # target tick missed, or the actor left
            truth = positions[actor]
            row = self.rows.setdefault(key, {h: {'ours': [], 'stay': [], 'raw': [], 'in95': []}
                                             for h in range(len(self.horizons))})[k]
            e = pred - truth
            row['ours'].append(float(np.hypot(*e)))
            row['stay'].append(float(np.hypot(*(stay - truth))))
            row['raw'].append(float(np.hypot(*(raw - truth))))
            if k == len(self.horizons) - 1:
                self.by_model.setdefault((key, model), []).append(
                    (row['ours'][-1], row['raw'][-1], row['stay'][-1]))
            try:
                row['in95'].append(float(e @ np.linalg.solve(cov, e)) <= CHI2_95_2D)
            except np.linalg.LinAlgError:
                pass
        self.pending = keep

    def summary(self):
        out = {}
        for key, per_h in sorted(self.rows.items()):
            out[key] = []
            for k, h in enumerate(self.horizons):
                r = per_h[k]
                n = len(r['ours'])
                if not n:
                    continue
                out[key].append({'h': h, 'n': n,
                                 'ours': float(np.mean(r['ours'])), 'ours_p95': float(np.percentile(r['ours'], 95)),
                                 'stay': float(np.mean(r['stay'])), 'raw': float(np.mean(r['raw'])),
                                 'in95': 100.0 * float(np.mean(r['in95'])) if r['in95'] else float('nan')})
        return out


def format_by_model(by_model, horizon):
    lines = ['  error at %.1f s by the model the predictor chose (ours / raw cv / stay):' % horizon]
    names = {0: 'static', 1: 'cv', 2: 'ctrv'}
    for ((g, motion), m), errs in sorted(by_model.items()):
        e = np.array(errs)
        lines.append('    %-10s %-7s %-6s n=%6d   %5.2f / %5.2f / %5.2f m'
                     % (g, motion, names[m], len(e), e[:, 0].mean(), e[:, 1].mean(), e[:, 2].mean()))
    return '\n'.join(lines)


def format_summary(s, models, title):
    lines = [title, '  class / truly   horizon      n | ours MAE  p95 | stay MAE  raw-cv MAE | in 95 % ellipse']
    for (g, motion), rows in s.items():
        for r in rows:
            lines.append('  %-10s %-7s %4.1f s %6d | %5.2f m %5.2f | %5.2f m   %5.2f m   |     %5.1f %%'
                         % (g, motion, r['h'], r['n'], r['ours'], r['ours_p95'], r['stay'], r['raw'], r['in95']))
    use = {}
    for (key, model), n in models.items():
        use.setdefault(key, {})[model] = n
    for (g, motion), m in sorted(use.items()):
        tot = sum(m.values())
        lines.append('  models used, %s %s: %s' % (g, motion, ', '.join(
            '%s %.0f %%' % (name, 100.0 * m.get(i, 0) / tot) for i, name in ((0, 'static'), (1, 'cv'), (2, 'ctrv')))))
    return '\n'.join(lines)
