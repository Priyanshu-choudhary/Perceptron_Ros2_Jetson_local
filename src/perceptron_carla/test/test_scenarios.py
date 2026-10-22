"""Tests for the Stage 9 suite's metric extraction and limits (scenarios.py), and the
bridge's --dropout parsing."""

import pytest

from perceptron_carla import scenarios as sc

SUMMARIES = {
    'cloud_eval_node': {'ground_precision': 0.991, 'ground_recall': 0.97, 'under_split_pct': 2.0},
    'fusion_eval_node': {
        'pedestrian 5-10 m': {'boxes': 100, 'coverage_pct': 90.0, 'range_mae': 0.1, 'correct_pct': 100.0},
        'pedestrian 10-20 m': {'boxes': 100, 'coverage_pct': 50.0, 'range_mae': 0.4, 'correct_pct': None},
        'vehicle 5-10 m': {'boxes': 10, 'coverage_pct': None, 'range_mae': None, 'correct_pct': None}},
    'tracking_eval_node': {'groups': {'pedestrian': {
        'pos_mae': 0.2, 'v_moving_mae': 0.15, 'v_static_mae': None, 'actor_frames': 400, 'id_switches': 2}}},
    'prediction_eval_node': {'pedestrian moving': [
        {'h': 1.0, 'ours': 0.35, 'stay': 0.95, 'in95': 92.0},
        {'h': 2.0, 'ours': 0.86, 'stay': 1.8, 'in95': 90.0}]},
    'costmap_eval_node': {'recall': {'pedestrian True': [500, 98.9], 'vehicle False': [80, 100.0]},
                          'false_obstacle_pct': 0.95, 'dynamic': {'1.0': [300, 95.0, 70.0], '2.0': [0, None, None]},
                          'flicker_pct': 0.0, 'ms_mean': 48.8, 'ms_p95': 54.0},
    'localization_eval_node': {'/odometry/global': {'pos_rmse_m': 1.1, 'head_rmse_deg': 2.0, 'samples': 900},
                               '/odometry/local': None},
}


def test_extract_metrics_flattens_every_evaluator():
    m = sc.extract_metrics(SUMMARIES)
    assert m['cloud.ground_precision'] == pytest.approx(99.1)
    assert m['fusion.pedestrian.coverage_pct'] == pytest.approx(70.0)          # box-weighted
    assert m['fusion.pedestrian.range_mae'] == pytest.approx((90 * 0.1 + 50 * 0.4) / 140)   # by estimates
    assert 'fusion.vehicle.coverage_pct' not in m                                # no values -> absent
    assert m['tracking.pedestrian.id_switches_per_100'] == pytest.approx(0.5)
    assert 'tracking.pedestrian.v_static_mae' not in m
    assert m['prediction.pedestrian_moving.err_2s'] == pytest.approx(0.86)
    assert m['prediction.pedestrian_moving.gain_vs_stay_2s'] == pytest.approx(100 * (1 - 0.86 / 1.8))
    assert m['costmap.recall.pedestrian_moving'] == 98.9 and m['costmap.recall.vehicle_static'] == 100.0
    assert m['costmap.dynamic_1s_pct'] == 95.0 and 'costmap.dynamic_2s_pct' not in m
    assert m['loc.global.pos_rmse_m'] == 1.1 and 'loc.local.pos_rmse_m' not in m


def test_check_and_verdict():
    m = {'a': 1.0, 'b': 5.0}
    rows = sc.check(m, [('a', '>=', 0.5), ('b', '<=', 4.0), ('c', '==', 0)])
    assert [r[4] for r in rows] == ['PASS', 'FAIL', 'MISSING']
    assert sc.verdict(rows) == 'FAIL'
    assert sc.verdict(rows[:1]) == 'PASS' and sc.verdict([rows[0], rows[2]]) == 'INCOMPLETE'


def test_every_scenario_limit_names_a_metric_the_suite_produces():
    produced = set(sc.extract_metrics(SUMMARIES)) | {
        'health.nodes_missing', 'health.outputs_after_dropout', 'rate.realtime_factor'}
    prefixes = ('cloud.', 'fusion.', 'tracking.', 'prediction.', 'costmap.', 'loc.', 'nav.')
    for name, spec in sc.SCENARIOS.items():
        assert spec['stack'] in sc.EVALUATORS
        for metric, op, _ in spec['limits']:
            assert op in sc.OPS
            assert metric in produced or metric.startswith(prefixes), (name, metric)


def test_common_limits_only_where_answerable():
    lim = sc.limits_for('crossing', {'rate.realtime_factor': 1.0})
    names = [n for n, _, _ in lim]
    assert 'rate.realtime_factor' in names and 'health.nodes_missing' not in names


def test_report_lists_failures():
    rows = sc.check({'x': 2.0}, [('x', '<=', 1.0)])
    text = sc.format_report([{'name': 's', 'about': 'a', 'verdict': 'FAIL', 'rows': rows,
                              'metrics': {'x': 2.0, 'y': 3.0}, 'notes': ['n']}], 'T')
    assert '| s | **FAIL** | x |' in text and '| y | 3 |' in text and '> n' in text


def test_dropout_spec_parsing():
    carla = pytest.importorskip('carla')       # noqa: F841  the bridge module imports carla
    from perceptron_carla.carla_bridge import parse_dropouts
    assert parse_dropouts('lidar:20:2,camera:35:1.5') == [('lidar', 20.0, 22.0), ('camera', 35.0, 36.5)]
    assert parse_dropouts('') == []
    with pytest.raises(ValueError):
        parse_dropouts('radar:1:1')


def test_rect_distance():
    a = sc.box_corners(0.0, 0.0, 0.0, -1.0, 1.0, -0.5, 0.5)
    b = sc.box_corners(3.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5)          # 1.5 m gap along x
    assert sc.rect_distance(a, b) == pytest.approx(1.5)
    c = sc.box_corners(1.2, 0.0, 0.7853981633974483, -0.5, 0.5, -0.5, 0.5)   # rotated, overlapping
    assert sc.rect_distance(a, c) == 0.0
    d = sc.box_corners(0.0, 2.0, 0.0, -0.2, 0.2, -0.2, 0.2)          # pedestrian beside the car
    assert sc.rect_distance(a, d) == pytest.approx(1.3)
    # the frame offset: a rear-axle frame box [-0.3, 2.1] seen from a car heading +y
    e = sc.box_corners(0.0, 0.0, 1.5707963267948966, -0.3, 2.1, -0.8, 0.8)
    assert e[:, 1].min() == pytest.approx(-0.3) and e[:, 1].max() == pytest.approx(2.1)


def test_nav_scenarios_are_complete():
    for name in ('nav_open', 'nav_cars', 'nav_people', 'nav_mixed'):
        spec = sc.SCENARIOS[name]
        assert spec['stack'] == 'navigation' and '--control ros' in spec['bridge']
        assert ('road' in spec['goal']) != ('ahead' in spec['goal'])
        names = [m for m, _, _ in spec['limits']]
        assert 'nav.reached' in names and 'nav.collisions' in names
