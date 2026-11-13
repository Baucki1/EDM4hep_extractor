import numpy as np
import pytest

import seed_utils as su

# d0, z0, phi0, tan_lambda, kappa, t0
BARREL = np.array([0.5, 3.0, 0.7, 0.4, 1e-3, 0.2])


def on_helix(params, s):
    s = np.asarray(s, dtype=float)
    norm = np.sqrt(1 + params[3] ** 2)
    return su.helix_position(params, s), params[5] + s * norm / su.C_MM_PER_NS


@pytest.mark.parametrize("kappa", [1e-3, -1e-3, 0.0])
def test_fit_recovers_exact_helix(kappa):
    params = BARREL.copy()
    params[4] = kappa
    pos, t = on_helix(params, [100.0, 200.0, 300.0])

    fitted = su.fit_helices(pos[None], t[None])[0]

    assert np.allclose(fitted, params, atol=1e-6)


def test_fit_without_time_leaves_t0_undefined():
    pos, _ = on_helix(BARREL, [100.0, 200.0, 300.0])

    assert np.isnan(su.fit_helices(pos[None])[0, 5])


@pytest.mark.parametrize("params", [BARREL, np.array([-0.3, -5.0, -2.0, 3.0, -5e-4, 0.0])])
def test_hits_on_the_helix_have_zero_residuals(params):
    s = np.array([50.0, 120.0, 400.0])
    pos, t = on_helix(params, s)

    res = su.helix_residuals(np.tile(params, (3, 1)), pos, t)

    assert np.allclose(res["s"], s, atol=1e-6)
    for key in ("d_perp", "d_n", "d_t", "dist"):
        assert np.allclose(res[key], 0.0, atol=1e-6)


def test_residuals_separate_perpendicular_normal_and_time_offsets():
    s = 250.0
    pos, t = on_helix(BARREL, s)
    ang = BARREL[2] + BARREL[4] * s
    e_perp = np.array([-np.sin(ang), np.cos(ang), 0.0])
    n2 = np.array([-BARREL[3] * np.cos(ang), -BARREL[3] * np.sin(ang), 1.0]) / np.sqrt(1 + BARREL[3] ** 2)

    res = su.helix_residuals(BARREL, pos + 0.3 * e_perp + -0.2 * n2, t + 0.05)

    assert np.isclose(res["d_perp"], 0.3) and np.isclose(res["d_n"], -0.2)
    assert np.isclose(res["d_t"], 0.05) and np.isclose(res["s"], s)


def test_residuals_without_time_are_nan():
    pos, _ = on_helix(BARREL, 100.0)

    assert np.isnan(su.helix_residuals(BARREL, pos)["d_t"])


def make_event(rng, n_bib=300):
    particles = [
        np.array([0.0, 0.0, 0.0, 0.5, 8e-4, 0.0]),
        np.array([0.0, 1.0, 2.5, 3.0, -6e-4, 0.0]),
        np.array([0.2, -2.0, -1.5, -0.8, 2e-4, 0.0]),
        np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0]),
    ]
    xyz, pid, t = [], [], []
    s = np.linspace(60.0, 500.0, 9)
    for i, params in enumerate(particles):
        pos, time = on_helix(params, s)
        xyz.append(pos)
        t.append(time)
        pid.append(np.full(len(s), i + 1))
    xyz.append(np.column_stack([rng.uniform(-500, 500, (n_bib, 2)), rng.uniform(-1500, 1500, n_bib)]))
    t.append(rng.uniform(0, 5, n_bib))
    pid.append(np.zeros(n_bib, dtype=int))
    return particles, np.concatenate(xyz), np.concatenate(pid), np.concatenate(t)


def test_primary_and_duplicate_seeds_are_true_and_match_the_truth_helix():
    rng = np.random.default_rng(1)
    particles, xyz, pid, t = make_event(rng)

    seeds = su.generate_seeds(xyz, pid, t, n_duplicates=1, fake_ratio=0.0, rng=rng)

    assert (seeds["origin"] == su.ORIGIN_TRUE).sum() == 4
    assert (seeds["origin"] == su.ORIGIN_DUPLICATE).sum() == 4
    assert seeds["is_true"].all() and (seeds["purity"] == 1.0).all()
    for params, particle, hits in zip(seeds["params"], seeds["particle_id"], seeds["hits"]):
        assert (pid[hits] == particle).all()
        assert np.allclose(params, particles[particle - 1], atol=1e-5)
    primary = seeds["hits"][seeds["origin"] == su.ORIGIN_TRUE]
    duplicate = seeds["hits"][seeds["origin"] == su.ORIGIN_DUPLICATE]
    assert not any((p == d).all() for p in primary for d in duplicate)


def test_fake_seeds_are_labelled_by_purity():
    rng = np.random.default_rng(2)
    _, xyz, pid, t = make_event(rng)

    seeds = su.generate_seeds(xyz, pid, t, n_duplicates=0, fake_ratio=20.0, rng=rng)

    fake = seeds["origin"] == su.ORIGIN_FAKE
    assert fake.sum() > 0
    expected = (seeds["purity"] >= 1.0 - 1e-9) & (seeds["particle_id"] > 0)
    assert (seeds["is_true"][fake] == expected[fake]).all()
    assert (seeds["purity"][fake] < 1.0).any()


def test_min_purity_controls_what_counts_as_true():
    pid = np.array([1, 1, 2, 0, 0, 0, 3, 3, 3])
    triplets = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 4]])

    _, purity, strict = su._label_triplets(pid, triplets, 1.0)
    majority, _, loose = su._label_triplets(pid, triplets, 0.6)

    assert np.allclose(purity, [2 / 3, 0.0, 1.0, 1 / 3])
    assert strict.tolist() == [False, False, True, False]
    assert loose.tolist() == [True, False, True, False]
    assert majority.tolist() == [1, 0, 3, 1]


def road_setup():
    rng = np.random.default_rng(3)
    particles, xyz, pid, t = make_event(rng, n_bib=0)
    behind, _ = on_helix(particles[0], [-100.0])
    xyz = np.vstack([xyz, behind])
    pid = np.append(pid, 0)
    seeds = su.generate_seeds(xyz, pid, n_duplicates=0, fake_ratio=0.0, rng=rng)
    return xyz, pid, seeds


def test_road_contains_the_own_track_and_nothing_behind_the_perigee():
    xyz, pid, seeds = road_setup()

    road = su.build_road_edges(seeds["params"], xyz, max_perp=2.0, max_n=5.0)
    labels = su.hit_seed_labels(seeds["particle_id"], seeds["is_true"], pid, road["seed"], road["hit"])

    assert len(xyz) - 1 not in road["hit"]
    assert labels.all()
    for i, particle in enumerate(seeds["particle_id"]):
        assert set(np.nonzero(pid == particle)[0]) == set(road["hit"][road["seed"] == i])


def test_road_cap_and_chunking():
    xyz, _, seeds = road_setup()

    capped = su.build_road_edges(seeds["params"], xyz, max_perp=2.0, max_n=5.0, max_per_seed=3)
    one = su.build_road_edges(seeds["params"], xyz, max_perp=2.0, max_n=5.0, chunk_size=1)
    many = su.build_road_edges(seeds["params"], xyz, max_perp=2.0, max_n=5.0, chunk_size=64)

    assert np.bincount(capped["seed"]).max() == 3
    assert set(zip(one["seed"], one["hit"])) == set(zip(many["seed"], many["hit"]))


def test_hit_seed_labels_reject_fake_seeds_and_bib_hits():
    labels = su.hit_seed_labels(
        seed_particle=[1, 1], seed_is_true=[True, False], hit_particle=[1, 0, 1],
        seed_idx=[0, 0, 1], hit_idx=[0, 1, 2],
    )

    assert labels.tolist() == [1, 0, 0]
