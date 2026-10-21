"""Tests for the Stage 8 rolling costmap (costmap.py): one per layer behaviour."""

import numpy as np

from perceptron_perception import costmap as cm

SENSOR = np.array([0.0, 0.0, 0.4])


def wall(x, y0, y1, z0=0.1, z1=0.4, step=0.1):
    ys, zs = np.meshgrid(np.arange(y0, y1, step), np.arange(z0, z1, step))
    return np.column_stack([np.full(ys.size, x), ys.ravel(), zs.ravel()])


def ground(x0, x1, y0, y1, step=0.25):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])


def fresh(**kw):
    m = cm.RollingCostmap(size=20.0, res=0.2, **kw)
    m.recenter((0.0, 0.0))
    return m


def cell(m, x, y):
    (i, j), _ = m.cell_of(np.array([x, y]))
    return i, j


def test_obstacle_becomes_lethal_and_ground_stays_free():
    m = fresh()
    m.integrate(wall(5.0, -1.0, 1.0), ground(1.0, 4.5, -1, 1), SENSOR, 0.0)
    lethal = m.lethal()
    assert lethal[cell(m, 5.05, 0.0)]
    assert not lethal[cell(m, 3.0, 0.0)]
    assert m.known()[cell(m, 3.0, 0.0)] and not m.known()[cell(m, -8.0, -8.0)]


def test_height_band_ignores_bumps_and_overhangs():
    m = fresh(min_obstacle_height=0.05, max_obstacle_height=0.8)
    pts = np.array([[4.0, 0.0, 0.02], [6.0, 0.0, 1.5], [8.0, 0.0, 0.5]])
    m.integrate(pts, [], SENSOR, 0.0)
    lethal = m.lethal()
    assert not lethal[cell(m, 4.0, 0.0)]                 # wheels roll over it
    assert not lethal[cell(m, 6.0, 0.0)]                 # robot passes under it
    assert lethal[cell(m, 8.0, 0.0)]


def test_rays_clear_an_obstacle_that_left():
    m = fresh()
    m.integrate(wall(5.0, -1.0, 1.0), [], SENSOR, 0.0)
    assert m.lethal()[cell(m, 5.05, 0.0)]
    for k in range(1, 6):                                # it drove off: lidar now sees the
        m.integrate(wall(8.0, -1.0, 1.0), ground(1.0, 7.5, -1.2, 1.2), SENSOR, 0.05 * k)  # ground + wall behind
    assert not m.lethal()[cell(m, 5.05, 0.0)]
    assert m.lethal()[cell(m, 8.05, 0.0)]


def test_unseen_obstacle_decays():
    m = fresh(decay_time=1.0)
    m.integrate(wall(5.0, -1.0, 1.0), [], SENSOR, 0.0)
    m.integrate(np.zeros((0, 3)), [], SENSOR, 2.0)       # 2 s, nothing seen there
    assert not m.lethal()[cell(m, 5.05, 0.0)]


def test_moving_object_points_are_excluded():
    m = fresh()
    m.integrate(wall(5.0, -0.3, 0.3), [], SENSOR, 0.0, exclude=[((5.0, 0.0), 0.6)])
    assert not m.lethal().any()


def test_recentering_keeps_obstacles_in_place():
    m = fresh()
    m.integrate(wall(5.0, -1.0, 1.0), [], SENSOR, 0.0)
    m.recenter((6.0, 0.0))                               # robot drove 6 m: the window shifts
    assert abs(m.origin[0] - (-4.0)) < 1e-9
    assert m.lethal()[cell(m, 5.05, 0.0)]


def test_inflation_follows_nav2_convention():
    m = fresh(inscribed_radius=0.2, inflation_radius=1.0, cost_scaling=2.5)
    lethal = np.zeros((m.nx, m.ny), bool)
    lethal[50, 50] = True
    cost = m.inflate(lethal)
    assert cost[50, 50] == cm.LETHAL
    assert cost[51, 50] == cm.INSCRIBED                  # 0.2 m away
    assert 0 < cost[54, 50] < 252                         # 0.8 m away
    assert cost[60, 50] == 0                              # beyond the inflation radius


def test_dynamic_risk_paints_the_path_ahead_not_behind():
    m = fresh(inscribed_radius=0.2)
    path = np.array([[6.0, -2.0 + 1.2 * 0.1 * k] for k in range(21)])    # crossing at 1.2 m/s
    ob = {'path': path, 'path_dt': 0.1, 'horizons': [0.5, 1.0, 2.0],
          'covs': np.array([np.eye(2) * 0.02, np.eye(2) * 0.05, np.eye(2) * 0.2]), 'radius': 0.4}
    # robot at 3 m/s reaches x = 6 in ~2 s: exactly when the pedestrian gets to y = 0.4
    risk, lethal = m.dynamic_risk([ob], robot_xy=(0.0, 0.0), robot_speed=3.0, nominal_speed=1.0)
    assert lethal[cell(m, 6.0, -2.0)]                    # where it is now
    assert risk[cell(m, 6.0, 0.4)] > 0.5                 # where it will be when we get there
    assert risk[cell(m, 6.0, -4.0)] < 0.05               # where it came from
    cost = cm.risk_to_cost(risk, lethal)
    assert cost[cell(m, 6.0, -2.0)] == cm.LETHAL and cost.max() == cm.LETHAL


def test_occupancy_grid_encoding():
    cost = np.array([[0, 254], [253, 126]], np.uint8)
    known = np.array([[False, True], [True, True]])
    occ = cm.to_occupancy(cost, known)
    assert occ.tolist() == [[-1, 100], [100, 50]]        # transposed: rows run along y


def test_dynamic_risk_fades_where_the_robot_cannot_meet_it():
    m = fresh(inscribed_radius=0.2)
    path = np.array([[6.0, -2.0 + 1.2 * 0.1 * k] for k in range(21)])
    ob = {'path': path, 'path_dt': 0.1, 'horizons': [0.5, 1.0, 2.0],
          'covs': np.array([np.eye(2) * 0.02, np.eye(2) * 0.05, np.eye(2) * 0.2]), 'radius': 0.4}
    # a 1 m/s robot needs ~6 s to get to x = 6; by then the pedestrian is long gone
    risk, _ = m.dynamic_risk([ob], robot_xy=(0.0, 0.0), robot_speed=0.0, nominal_speed=1.0)
    assert risk[cell(m, 6.0, 0.4)] < 0.15
    assert risk[cell(m, 6.0, -1.6)] > 0.5                # the next ~0.3 s is costly regardless


def test_ground_seen_where_a_car_was_clears_it():
    m = fresh()
    car = wall(6.0, -1.0, 1.0, z0=0.3, z1=1.2)            # body only: rays to the road pass under it
    m.integrate(car, [], SENSOR, 0.0)
    assert m.lethal()[cell(m, 6.05, 0.0)]
    for k in range(1, 4):                                  # car gone: the lidar now sees road there
        m.integrate(np.zeros((0, 3)), ground(5.0, 7.0, -1.0, 1.0, step=0.1), SENSOR, 0.05 * k)
    assert not m.lethal()[cell(m, 6.05, 0.0)]


def test_parked_car_survives_ground_clearing():
    m = fresh()
    side = wall(6.0, -1.0, 1.0, z0=0.2, z1=0.5)
    for k in range(5):                                     # its side keeps being hit
        m.integrate(side, ground(4.0, 5.8, -1.0, 1.0, step=0.1), SENSOR, 0.05 * k)
    assert m.lethal()[cell(m, 6.05, 0.0)]
