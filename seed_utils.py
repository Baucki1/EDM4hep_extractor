"""Helix seeds from truth tracklets (true, duplicate and fake), plus helix-to-hit residuals.

Seed parameters are the perigee with respect to the origin, in mm and ns: (d0, z0, phi0, tan_lambda,
kappa, t0). kappa = 1/R is signed, positive meaning counter-clockwise in xy along the direction of
travel. The field only enters when converting a momentum to kappa, so everything here is geometric.
"""
import numpy as np

C_MM_PER_NS = 299.792458
PARAM_NAMES = ("d0", "z0", "phi0", "tan_lambda", "kappa", "t0")
KAPPA_STRAIGHT = 1e-9
ORIGIN_TRUE, ORIGIN_DUPLICATE, ORIGIN_FAKE = 0, 1, 2


def _unpack(params):
    p = np.asarray(params, dtype=float)
    return [p[..., i] for i in range(6)]


def helix_position(params, s):
    """Position at transverse arc length s from the perigee; params and s broadcast."""
    d0, z0, phi0, tan_l, kappa, _ = _unpack(params)
    s = np.asarray(s, dtype=float)
    straight = np.abs(kappa) < KAPPA_STRAIGHT
    k = np.where(straight, 1.0, kappa)
    xp, yp = -d0 * np.sin(phi0), d0 * np.cos(phi0)
    ang = phi0 + k * s
    x = np.where(straight, xp + s * np.cos(phi0), xp + (np.sin(ang) - np.sin(phi0)) / k)
    y = np.where(straight, yp + s * np.sin(phi0), yp - (np.cos(ang) - np.cos(phi0)) / k)
    return np.stack([x, y, z0 + s * tan_l], axis=-1)


def fit_helices(xyz, t=None):
    """Helix through three hits ordered from inside out: xyz (K,3,3), t (K,3) or None -> params (K,6).

    t0 is NaN without hit times.
    """
    xyz = np.asarray(xyz, dtype=float)
    p1, p2, p3 = xyz[:, 0, :2], xyz[:, 1, :2], xyz[:, 2, :2]
    a, b = p2 - p1, p3 - p1
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    la, lb, lc = np.hypot(*a.T), np.hypot(*b.T), np.hypot(*(b - a).T)
    denom = la * lb * lc
    kappa = np.sign(cross) * np.where(denom > 0, 2 * np.abs(cross) / np.where(denom > 0, denom, 1), 0.0)
    straight = np.abs(kappa) < KAPPA_STRAIGHT

    d2a, d2b = (a**2).sum(1), (b**2).sum(1)
    det = np.where(straight, 1.0, 2 * cross)
    center = p1 + np.stack([(b[:, 1] * d2a - a[:, 1] * d2b) / det, (a[:, 0] * d2b - b[:, 0] * d2a) / det], 1)
    radius = 1.0 / np.where(straight, 1.0, np.abs(kappa))
    c_norm = np.hypot(*center.T)
    c_hat = np.where((c_norm > 1e-12)[:, None], center / np.where(c_norm > 1e-12, c_norm, 1)[:, None], [1.0, 0.0])
    peri_curved = center - radius[:, None] * c_hat
    dir_curved = np.sign(kappa)[:, None] * np.stack([c_hat[:, 1], -c_hat[:, 0]], 1)

    line = p3 - p1
    dir_line = line / np.maximum(np.hypot(*line.T), 1e-12)[:, None]
    peri_line = p1 - (p1 * dir_line).sum(1)[:, None] * dir_line

    direction = np.where(straight[:, None], dir_line, dir_curved)
    peri = np.where(straight[:, None], peri_line, peri_curved)
    phi0 = np.arctan2(direction[:, 1], direction[:, 0])
    d0 = -peri[:, 0] * np.sin(phi0) + peri[:, 1] * np.cos(phi0)

    rel_c = xyz[:, :, :2] - center[:, None, :]
    pc = (peri - center)[:, None, :]
    pc = pc / np.maximum(np.hypot(pc[..., 0], pc[..., 1]), 1e-12)[..., None]
    theta = np.arctan2(pc[..., 0] * rel_c[..., 1] - pc[..., 1] * rel_c[..., 0], (pc * rel_c).sum(-1))
    s_line = ((xyz[:, :, :2] - peri[:, None, :]) * dir_line[:, None, :]).sum(-1)
    s = np.where(straight[:, None], s_line, theta / np.where(straight, 1.0, kappa)[:, None])

    s_mean, z_mean = s.mean(1), xyz[:, :, 2].mean(1)
    var = ((s - s_mean[:, None]) ** 2).mean(1)
    cov = ((s - s_mean[:, None]) * (xyz[:, :, 2] - z_mean[:, None])).mean(1)
    tan_l = np.where(var > 1e-12, cov / np.where(var > 1e-12, var, 1), 0.0)
    z0 = z_mean - tan_l * s_mean
    if t is None:
        t0 = np.full(len(xyz), np.nan)
    else:
        t0 = (np.asarray(t, dtype=float) - s * np.sqrt(1 + tan_l**2)[:, None] / C_MM_PER_NS).mean(1)
    return np.stack([d0, z0, phi0, tan_l, kappa, t0], axis=1)


def helix_residuals(params, pos, t=None, newton_steps=3):
    """Closest approach of hits to helices; params (...,6) and pos (...,3) broadcast.

    Returns s (arc length of the closest point), d_perp (in the xy plane, left of the direction of
    travel), d_n (in the plane spanned by the tangent and z, normal to it), d_t (hit time minus
    expected time, NaN without t) and dist (3D distance).
    """
    p = np.asarray(params, dtype=float)
    d0, z0, phi0, tan_l, kappa, t0 = _unpack(p)
    pos = np.asarray(pos, dtype=float)
    hx, hy, hz = pos[..., 0], pos[..., 1], pos[..., 2]
    straight = np.abs(kappa) < KAPPA_STRAIGHT
    k = np.where(straight, 1.0, kappa)
    xp, yp = -d0 * np.sin(phi0), d0 * np.cos(phi0)

    cx, cy = xp - np.sin(phi0) / k, yp + np.cos(phi0) / k
    w_x, w_y = np.sign(k) * np.sin(phi0), -np.sign(k) * np.cos(phi0)
    rx, ry = hx - cx, hy - cy
    s_xy = np.where(
        straight,
        (hx - xp) * np.cos(phi0) + (hy - yp) * np.sin(phi0),
        np.arctan2(w_x * ry - w_y * rx, w_x * rx + w_y * ry) / k,
    )
    s_z = np.where(np.abs(tan_l) > 1e-9, (hz - z0) / np.where(np.abs(tan_l) > 1e-9, tan_l, 1.0), s_xy)

    def dist2(s):
        return ((helix_position(p, s) - pos) ** 2).sum(-1)

    s = np.where(dist2(s_z) < dist2(s_xy), s_z, s_xy)
    kap = np.where(straight, 0.0, kappa)
    for _ in range(newton_steps):
        f = helix_position(p, s) - pos
        ang = phi0 + kap * s
        tang = np.stack([np.cos(ang), np.sin(ang), tan_l * np.ones_like(ang)], -1)
        curv = np.stack([-np.sin(ang) * kap, np.cos(ang) * kap, np.zeros_like(ang)], -1)
        t2 = (tang * tang).sum(-1)
        den = t2 + (f * curv).sum(-1)
        s = s - (f * tang).sum(-1) / np.where(den > 1e-6, den, t2)

    ang = phi0 + kap * s
    e_perp = np.stack([-np.sin(ang), np.cos(ang), np.zeros_like(ang)], -1)
    norm_t = np.sqrt(1 + tan_l**2)
    n2 = np.stack([-tan_l * np.cos(ang), -tan_l * np.sin(ang), np.ones_like(ang)], -1) / norm_t[..., None]
    r = pos - helix_position(p, s)
    d_t = np.full(s.shape, np.nan) if t is None else np.asarray(t, dtype=float) - (t0 + s * norm_t / C_MM_PER_NS)
    return {
        "s": s,
        "d_perp": (r * e_perp).sum(-1),
        "d_n": (r * n2).sum(-1),
        "d_t": d_t,
        "dist": np.sqrt((r**2).sum(-1)),
    }


def build_road_edges(params, hit_xyz, hit_t=None, max_perp=5.0, max_n=10.0, min_s=0.0,
                     max_per_seed=None, chunk_size=64):
    """Candidate (seed, hit) pairs whose closest-approach residuals are inside a window.

    Hits behind the perigee (s < min_s) are dropped. With max_per_seed, only the hits with the
    smallest normalised residual are kept per seed. Returns flat arrays seed, hit, s, d_perp, d_n, d_t.
    """
    params = np.asarray(params, dtype=float)
    hit_xyz = np.asarray(hit_xyz, dtype=float)
    d0, _, phi0, _, kappa, _ = _unpack(params)
    straight = np.abs(kappa) < KAPPA_STRAIGHT
    k = np.where(straight, 1.0, kappa)
    xp, yp = -d0 * np.sin(phi0), d0 * np.cos(phi0)
    cx, cy, radius = xp - np.sin(phi0) / k, yp + np.cos(phi0) / k, 1.0 / np.abs(k)

    out = {key: [] for key in ("seed", "hit", "s", "d_perp", "d_n", "d_t")}
    for start in range(0, len(params), chunk_size):
        sl = slice(start, start + chunk_size)
        hx, hy = hit_xyz[None, :, 0], hit_xyz[None, :, 1]
        # In-plane distance to the circle approximates |d_perp|; the margin absorbs the 3D refinement.
        circ = np.where(
            straight[sl, None],
            np.abs((hx - xp[sl, None]) * np.sin(phi0[sl, None]) - (hy - yp[sl, None]) * np.cos(phi0[sl, None])),
            np.abs(np.hypot(hx - cx[sl, None], hy - cy[sl, None]) - radius[sl, None]),
        )
        rows, cols = np.nonzero(circ < 1.5 * max_perp)
        res = helix_residuals(params[sl][rows], hit_xyz[cols], None if hit_t is None else np.asarray(hit_t)[cols])
        keep = (np.abs(res["d_perp"]) < max_perp) & (np.abs(res["d_n"]) < max_n) & (res["s"] >= min_s)
        rows, cols = rows[keep], cols[keep]
        res = {key: val[keep] for key, val in res.items()}
        if max_per_seed is not None and len(rows):
            score = np.hypot(res["d_perp"] / max_perp, res["d_n"] / max_n)
            order = np.lexsort((score, rows))
            rows_sorted = rows[order]
            first = np.searchsorted(rows_sorted, rows_sorted, side="left")
            order = order[np.arange(len(order)) - first < max_per_seed]
            rows, cols = rows[order], cols[order]
            res = {key: val[order] for key, val in res.items()}
        out["seed"].append(rows + start)
        out["hit"].append(cols)
        for key in ("s", "d_perp", "d_n", "d_t"):
            out[key].append(res[key])
    return {key: np.concatenate(val) if val else np.empty(0) for key, val in out.items()}


def _label_triplets(particle_id, triplets, min_purity):
    """Majority particle, purity and truth flag of hit triplets; particle 0 (BIB/noise) never matches."""
    pid = particle_id[triplets]
    same = (pid[:, :, None] == pid[:, None, :]).sum(2) * (pid > 0)
    best = same.argmax(1)
    majority_count = same.max(1)
    majority = np.where(majority_count > 0, pid[np.arange(len(pid)), best], 0)
    purity = majority_count / 3.0
    return majority, purity, (majority > 0) & (purity >= min_purity - 1e-9)


def _tracklets(particle_id, radius, n_duplicates, max_start, rng):
    idx = np.nonzero(particle_id > 0)[0]
    order = idx[np.lexsort((radius[idx], particle_id[idx]))]
    _, first, count = np.unique(particle_id[order], return_index=True, return_counts=True)
    primary, duplicate = [], []
    for f, n in zip(first, count):
        if n < 3:
            continue
        pick = rng.permutation(min(n - 2, max_start))[: 1 + n_duplicates]
        rows = order[f + pick[:, None] + np.arange(3)]
        primary.append(rows[0])
        duplicate.extend(rows[1:])
    empty = np.empty((0, 3), dtype=np.int64)
    return (np.array(primary) if primary else empty), (np.array(duplicate) if duplicate else empty)


def _pick_outward(tree, points, radius, r_min, k, min_dr, rng):
    _, nbr = tree.query(points, k=k)
    valid = radius[nbr] > (r_min + min_dr)[:, None]
    best = np.where(valid, rng.random(nbr.shape), -1.0).argmax(1)
    return np.where(valid.any(1), nbr[np.arange(len(nbr)), best], -1)


def _fake_triplets(xyz, radius, n_fake, neighbors, min_dr, rng):
    from scipy.spatial import cKDTree

    k = min(neighbors, len(xyz))
    tree = cKDTree(xyz)
    a = rng.integers(0, len(xyz), n_fake)
    b = _pick_outward(tree, xyz[a], radius, radius[a], k, min_dr, rng)
    a, b = a[b >= 0], b[b >= 0]
    c = _pick_outward(tree, 2 * xyz[b] - xyz[a], radius, radius[b], k, min_dr, rng)
    return np.stack([a[c >= 0], b[c >= 0], c[c >= 0]], 1)


def generate_seeds(xyz, particle_id, t=None, n_duplicates=1, fake_ratio=5.0, max_start=3,
                   min_purity=1.0, fake_neighbors=16, min_dr=1.0, rng=None):
    """Helix seeds from three consecutive hits, with labels.

    xyz (N,3) and particle_id (N,) describe one event, particle 0 being BIB/noise. Every particle with at
    least three hits gets one primary tracklet and up to n_duplicates further ones, all starting in its
    first max_start triplets (inside out by distance from the origin). fake_ratio fake seeds per
    primary one are drawn from nearby outward hits, so fewer may come out in sparse regions.

    A seed counts as true when its majority particle covers at least min_purity of its three hits;
    the default 1.0 needs all three. Returns params (K,6), hits (K,3) row indices into xyz, particle_id
    (majority particle, 0 if none), purity, is_true and origin (ORIGIN_TRUE / _DUPLICATE / _FAKE).
    """
    rng = np.random.default_rng() if rng is None else rng
    xyz = np.asarray(xyz, dtype=float)
    particle_id = np.asarray(particle_id).astype(np.int64)
    radius = np.linalg.norm(xyz, axis=1)

    primary, duplicate = _tracklets(particle_id, radius, n_duplicates, max_start, rng)
    n_fake = int(round(fake_ratio * len(primary)))
    if n_fake and len(xyz) >= 3:
        fake = _fake_triplets(xyz, radius, n_fake, fake_neighbors, min_dr, rng)
    else:
        fake = np.empty((0, 3), dtype=np.int64)

    hits = np.concatenate([primary, duplicate, fake]).astype(np.int64)
    origin = np.concatenate(
        [np.full(len(primary), ORIGIN_TRUE), np.full(len(duplicate), ORIGIN_DUPLICATE), np.full(len(fake), ORIGIN_FAKE)]
    )
    params = fit_helices(xyz[hits], None if t is None else np.asarray(t)[hits])
    finite = np.isfinite(params[:, :5]).all(1)
    hits, origin, params = hits[finite], origin[finite], params[finite]
    majority, purity, is_true = _label_triplets(particle_id, hits, min_purity)
    return {
        "params": params,
        "hits": hits,
        "particle_id": majority,
        "purity": purity,
        "is_true": is_true,
        "origin": origin,
    }


def hit_seed_labels(seed_particle, seed_is_true, hit_particle, seed_idx, hit_idx):
    """1 where the hit belongs to the particle of a true seed; fake seeds and BIB hits are always 0."""
    seed_idx, hit_idx = np.asarray(seed_idx), np.asarray(hit_idx)
    same = np.asarray(seed_particle)[seed_idx] == np.asarray(hit_particle)[hit_idx]
    return (same & np.asarray(seed_is_true)[seed_idx] & (np.asarray(hit_particle)[hit_idx] > 0)).astype(np.int64)
