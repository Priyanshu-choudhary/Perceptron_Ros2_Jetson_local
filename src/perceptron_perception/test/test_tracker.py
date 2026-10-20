"""Tests for the Stage 6 tracker on synthetic motion (tracker.py)."""

import math

import numpy as np

from perceptron_perception import tracker as tk

DT = 0.05


def det(trk, x, y, cls='pedestrian', det_id='', noise=0.0, rng=None):
    if rng is not None and noise:
        x, y = x + rng.normal(0, noise), y + rng.normal(0, noise)
    return trk.make_detection((x, y), 0.9, cls, np.eye(2) * 0.01 ** 2, (0.5, 0.5, 1.8), det_id)


def confirmed(trk):
    return [t for t in trk.tracks if t.state != tk.TENTATIVE]


def test_walking_person_speed_converges_and_keeps_one_id():
    trk, rng = tk.Tracker(), np.random.default_rng(0)
    for k in range(60):                                          # 3 s at 1.2 m/s along +y
        trk.step([det(trk, 10.0, -2.0 + 1.2 * k * DT, noise=0.05, rng=rng)], k * DT)
    c = confirmed(trk)
    assert len(c) == 1 and c[0].hits == 60
    assert abs(c[0].x[3] - 1.2) < 0.15 and abs(c[0].x[2]) < 0.15
    assert c[0].moving(trk.min_speed)


def test_two_people_crossing_paths_keep_their_ids():
    trk, rng = tk.Tracker(), np.random.default_rng(1)
    ids = None
    for k in range(80):                                          # A goes +y, B goes -y, 1 m apart in x
        t = k * DT
        trk.step([det(trk, 10.0, -2.0 + 1.0 * t, det_id='A', noise=0.03, rng=rng),
                  det(trk, 11.0, 2.0 - 1.0 * t, det_id='B', noise=0.03, rng=rng)], t)
        now = {tr.source: tr.id for tr in confirmed(trk)}
        if ids is None and len(now) == 2:
            ids = now
        if ids is not None:
            assert now == ids, (k, now, ids)
    assert ids is not None


def test_short_occlusion_keeps_id_long_one_does_not():
    trk = tk.Tracker(max_coast_time=1.0)
    t, k = 0.0, 0
    for k in range(20):
        t = k * DT
        trk.step([det(trk, 8.0, 1.0 * t)], t)
    first = confirmed(trk)[0].id
    for _ in range(10):                                          # 0.5 s behind a pole
        k += 1
        t = k * DT
        trk.step([], t)
        assert confirmed(trk)[0].state == tk.COASTING
    k += 1
    t = k * DT
    trk.step([det(trk, 8.0, 1.0 * t)], t)
    assert confirmed(trk)[0].id == first                         # reappears where predicted
    for _ in range(30):                                          # 1.5 s gone: deleted
        k += 1
        trk.step([], k * DT)
    assert not trk.tracks


def test_single_flicker_never_confirms():
    trk = tk.Tracker()
    trk.step([det(trk, 5.0, 5.0)], 0.0)
    for k in range(1, 5):
        trk.step([], k * DT)
    assert not confirmed(trk) and not trk.tracks


def test_parked_car_is_not_moving():
    trk, rng = tk.Tracker(), np.random.default_rng(2)
    for k in range(60):
        trk.step([det(trk, 15.0, 3.0, cls='car', noise=0.2, rng=rng)], k * DT)
    c = confirmed(trk)[0]
    assert c.speed < 0.5 and not c.moving(trk.min_speed)


def test_class_gating():
    trk = tk.Tracker()
    for k in range(5):
        trk.step([det(trk, 10.0, 0.0)], k * DT)
    trk.step([det(trk, 10.0, 0.0, cls='car')], 5 * DT)
    groups = sorted(t.group for t in trk.tracks)
    assert groups == ['pedestrian', 'vehicle']


def test_near_face_correction_pushes_away_from_sensor():
    xy = tk.correct_near_face((10.0, 0.0), (0.0, 0.0), 1.0)
    assert np.allclose(xy, (11.0, 0.0))
    xy = tk.correct_near_face((3.0, 4.0), (0.0, 0.0), 1.0)
    assert math.isclose(np.linalg.norm(xy), 6.0)


def test_instant_turnaround_keeps_one_id():
    # the --crossing pedestrians reverse in a single tick: +1.4 m/s -> -1.4 m/s
    trk, rng = tk.Tracker(), np.random.default_rng(3)
    y, v, ids = -4.0, 1.4, set()
    for k in range(200):
        y += v * DT
        if y > 4.0 or y < -4.0:
            v = -v
        trk.step([det(trk, 10.0, y, noise=0.03, rng=rng)], k * DT)
        ids |= {t.id for t in confirmed(trk)}
    assert len(ids) == 1, ids
    assert len(trk.tracks) == 1


def test_duplicate_tracks_are_pruned():
    trk = tk.Tracker()
    a = trk.make_detection((10.0, 0.0), 0.9, 'pedestrian', np.eye(2) * 1e-4, (0.5, 0.5, 1.8))
    b = trk.make_detection((10.2, 0.0), 0.9, 'pedestrian', np.eye(2) * 1e-4, (0.5, 0.5, 1.8))
    tracks = [tk.Track(a, 0.0, tk.DEFAULT_CLASS_PARAMS['pedestrian'], 4.0),
              tk.Track(b, 0.0, tk.DEFAULT_CLASS_PARAMS['pedestrian'], 4.0)]
    tracks[0].hits = 10
    kept = tk.Tracker._prune(tracks)
    assert [t.hits for t in kept] == [10]
