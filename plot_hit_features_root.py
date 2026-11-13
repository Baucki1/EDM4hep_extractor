import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import ROOT

# Reco tracker-hit collections and their hit<->sim-hit link collections
HIT_COLLECTIONS = {
    "VXDBarrelHits": "VXDBarrelHitsRelations",
    "ITBarrelHits": "ITBarrelHitsRelations",
    "OTBarrelHits": "OTBarrelHitsRelations",
    "VXDEndcapHits": "VXDEndcapHitsRelations",
    "ITEndcapHits": "ITEndcapHitsRelations",
    "OTEndcapHits": "OTEndcapHitsRelations",
}

# Reco-hit collection -> fired-pixel collection and hit->pixel relation collection
PIXEL_COLLECTIONS = {
    "VXDBarrelHits": "VXDBarrelPixels",
    "ITBarrelHits": "ITBarrelPixels",
    "OTBarrelHits": "OTBarrelPixels",
    "VXDEndcapHits": "VXDEndcapPixels",
    "ITEndcapHits": "ITEndcapPixels",
    "OTEndcapHits": "OTEndcapPixels",
}
RAW_REL_COLLECTIONS = {
    "VXDBarrelHits": "VXDBarrelRawHitRelations",
    "ITBarrelHits": "ITBarrelRawHitRelations",
    "OTBarrelHits": "OTBarrelRawHitRelations",
    "VXDEndcapHits": "VXDEndcapRawHitRelations",
    "ITEndcapHits": "ITEndcapRawHitRelations",
    "OTEndcapHits": "OTEndcapRawHitRelations",
}

# Reco-hit collection -> sim-hit collection (used to resolve SimTrackerHit -> MCParticle)
SIM_COLLECTIONS = {
    "VXDBarrelHits": "VertexBarrelCollection",
    "ITBarrelHits": "InnerTrackerBarrelCollection",
    "OTBarrelHits": "OuterTrackerBarrelCollection",
    "VXDEndcapHits": "VertexEndcapCollection",
    "ITEndcapHits": "InnerTrackerEndcapCollection",
    "OTEndcapHits": "OuterTrackerEndcapCollection",
}

# Hit categories (label, color). Classification, per reco hit:
#   noise      : no related SimTrackerHit (digitisation noise)
#   bib        : related SimTrackerHit without MCParticle (overlaid BIB)
#   sim        : MCParticle created in simulation (secondaries, mapped to background)
#   signal     : MCParticle from the generator (not created in simulation)
CATEGORIES = {
    "signal": ("Signal hits", "#1f77b4"),
    "sim": ("Sim-particle hits (secondaries)", "#2ca02c"),
    "bib": ("BIB hits", "#d62728"),
    "noise": ("Noise hits", "#7f7f7f"),
}

BASE_FEATURES = ["x", "y", "z", "r", "phi", "theta", "t", "edep", "du", "dv"]
CLUSTER_FEATURES = [
    "npix", "ext_u", "ext_v", "tan_u", "tan_v", "res_v", "pull_v", "pix_charge",
    "q_sum", "q_max", "q_mean", "q_std", "q_peak_frac", "q_entropy",
    "fill_factor", "ext_ratio", "t_span_pix", "t_std_pix", "pca_axis_ratio",
]
FEATURES = BASE_FEATURES + CLUSTER_FEATURES
LABELS = {
    "x": "x [mm]", "y": "y [mm]", "z": "z [mm]", "r": "r [mm]",
    "phi": "phi [rad]", "theta": "theta [rad]", "t": "hit time [ns]",
    "edep": "EDep [GeV]", "du": "du [mm]", "dv": "dv [mm]",
    "npix": "cluster size [pixels]",
    "ext_u": "cluster extent u [pixels]",
    "ext_v": "cluster extent v [pixels]",
    "tan_u": "tan(incidence angle u)",
    "tan_v": "tan(incidence angle v)",
    "res_v": "residual v [um]",
    "pull_v": "pull v = residual v / dv",
    "pix_charge": "charge per fired pixel [electrons]",
    "q_sum": "cluster charge sum [electrons]",
    "q_max": "max pixel charge in cluster [electrons]",
    "q_mean": "mean pixel charge in cluster [electrons]",
    "q_std": "std(pixel charge) in cluster [electrons]",
    "q_peak_frac": "peakness: q_max / q_sum",
    "q_entropy": "charge entropy",
    "fill_factor": "cluster fill factor = npix / (ext_u*ext_v)",
    "ext_ratio": "shape ratio = ext_v / ext_u",
    "t_span_pix": "pixel time span in cluster [ns]",
    "t_std_pix": "std(pixel time) in cluster [ns]",
    "pca_axis_ratio": "PCA axis ratio (major/minor)",
}
# Features that benefit from a logarithmic x axis
LOGX = {"edep", "pix_charge", "q_sum", "q_max", "q_mean", "q_std", "pca_axis_ratio"}
INT_FEATURES = {"npix", "ext_u", "ext_v"}
SIMHIT_REQUIRED_FEATURES = {"tan_u", "tan_v", "res_v", "pull_v"}


def _setup_plot_style():
    """Apply consistent plotting style for publication-like figures."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 170,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
    })

BIT_CREATED_IN_SIMULATION = 30


SURFACE_SUBDETS = {
    "VXDBarrelHits": "VXDBarrel",
    "VXDEndcapHits": "VXDEndcap",
    "ITBarrelHits": "ITBarrel",
    "ITEndcapHits": "ITEndcap",
    "OTBarrelHits": "OTBarrel",
    "OTEndcapHits": "OTEndcap",
}
SUBDETECTOR_ORDER = list(HIT_COLLECTIONS.keys())


def _has_branch(tree, name):
    return tree.GetBranch(name) is not None


def _get_branch_obj(tree, name):
    """Return branch object via attribute access, or None if not readable."""
    if not _has_branch(tree, name):
        return None
    try:
        return getattr(tree, name)
    except Exception:
        return None


def _safe_getattr(obj, names, default=np.nan):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) else value
    return default


def _get_xyz(hit):
    pos = _safe_getattr(hit, ["position", "getPosition"], default=None)
    if pos is None:
        return np.nan, np.nan, np.nan
    if hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z"):
        return float(pos.x), float(pos.y), float(pos.z)
    return float(pos[0]), float(pos[1]), float(pos[2])


def _normalize_cellid(value):
    """Normalize any cellID to unsigned 64-bit representation."""
    try:
        return int(value) & ((1 << 64) - 1)
    except Exception:
        return -1


def _get_cellid(obj):
    return _normalize_cellid(_safe_getattr(obj, ["cellID", "getCellID"], default=-1))


def _get_momentum_xyz(obj):
    mom = _safe_getattr(obj, ["momentum", "getMomentum"], default=None)
    if mom is None:
        return np.nan, np.nan, np.nan
    if hasattr(mom, "x") and hasattr(mom, "y") and hasattr(mom, "z"):
        return float(mom.x), float(mom.y), float(mom.z)
    return float(mom[0]), float(mom[1]), float(mom[2])


def _load_surfaces_map(path):
    """Read surfaces.csv into {(subdet, cellID): (U, V, N)}."""
    if path is None:
        return {}
    if not os.path.exists(path):
        raise RuntimeError(f"surfaces CSV does not exist: {path}")

    needed = {"subdet", "cellID", "ux", "uy", "uz", "vx", "vy", "vz", "nx", "ny", "nz"}
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            missing = sorted(needed - set(reader.fieldnames or []))
            raise RuntimeError(f"surfaces CSV missing required columns: {missing}")
        for row in reader:
            try:
                key = (str(row["subdet"]), _normalize_cellid(row["cellID"]))
                U = np.array([float(row["ux"]), float(row["uy"]), float(row["uz"])], dtype=float)
                V = np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])], dtype=float)
                N = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])], dtype=float)
            except Exception:
                continue
            out[key] = (U, V, N)
    return out


def _is_created_in_simulation(mc_particle):
    status = int(_safe_getattr(mc_particle, ["simulatorStatus", "getSimulatorStatus"], default=0))
    return ((status >> BIT_CREATED_IN_SIMULATION) & 1) == 1


def _classify_hit(sim_index, sim_to_mc, mc_particles):
    if sim_index is None or sim_index < 0:
        return "noise"

    if sim_to_mc is None or sim_index >= len(sim_to_mc):
        return "bib"

    mc_oid = sim_to_mc[sim_index]
    mc_index = int(_safe_getattr(mc_oid, ["index"], default=-1))
    if mc_index < 0 or mc_index >= len(mc_particles):
        return "bib"

    return "sim" if _is_created_in_simulation(mc_particles[mc_index]) else "signal"


def _relation_map(rel_from, rel_to):
    rel = {}
    if rel_from is None or rel_to is None:
        return rel
    n = min(len(rel_from), len(rel_to))
    for i in range(n):
        from_idx = int(_safe_getattr(rel_from[i], ["index"], default=-1))
        to_idx = int(_safe_getattr(rel_to[i], ["index"], default=-1))
        if from_idx >= 0 and from_idx not in rel:
            rel[from_idx] = to_idx
    return rel


def _relation_multimap(rel_from, rel_to):
    """Build one-to-many map {from_index: [to_index, ...]} for link collections."""
    rel = {}
    if rel_from is None or rel_to is None:
        return rel
    n = min(len(rel_from), len(rel_to))
    for i in range(n):
        from_idx = int(_safe_getattr(rel_from[i], ["index"], default=-1))
        to_idx = int(_safe_getattr(rel_to[i], ["index"], default=-1))
        if from_idx < 0 or to_idx < 0:
            continue
        rel.setdefault(from_idx, []).append(to_idx)
    return rel


def _build_hit_to_pixel_map(rel_from, rel_to, n_hits, n_pixels):
    """Infer orientation and return map: hit_index -> [pixel_index, ...]."""
    if rel_from is None or rel_to is None:
        return {}

    # Option A: from=hit, to=pixel
    a = _relation_multimap(rel_from, rel_to)
    valid_a = 0
    for h, pixs in a.items():
        if 0 <= h < n_hits:
            valid_a += sum(1 for p in pixs if 0 <= p < n_pixels)

    # Option B: from=pixel, to=hit  -> invert to hit->pixels
    b_raw = _relation_multimap(rel_to, rel_from)
    valid_b = 0
    for h, pixs in b_raw.items():
        if 0 <= h < n_hits:
            valid_b += sum(1 for p in pixs if 0 <= p < n_pixels)

    chosen = a if valid_a >= valid_b else b_raw

    # Final cleanup / bounds check
    out = {}
    for h, pixs in chosen.items():
        if not (0 <= h < n_hits):
            continue
        good = [p for p in pixs if 0 <= p < n_pixels]
        if good:
            out[h] = good
    return out


def _find_branch_ci(name, branch_names):
    """Case-insensitive branch lookup. Return exact branch name or None."""
    if name in branch_names:
        return name
    lut = {b.lower(): b for b in branch_names}
    return lut.get(name.lower())


def _relation_candidates(rel_name):
    """Best-effort candidate endpoint collection names for a relation collection."""
    out = {"from": [], "to": []}

    # Common pattern: <RecoCollection>Relations
    if rel_name.endswith("Relations"):
        base = rel_name[:-len("Relations")]
        out["from"].append(base)

    # Tracker raw-hit relation: <Prefix>RawHitRelations => <Prefix>Hits -> <Prefix>Pixels
    if rel_name.endswith("RawHitRelations"):
        base = rel_name[:-len("RawHitRelations")]
        out["from"].append(base + "Hits")
        out["to"].append(base + "Pixels")

    # Tracker mapping used by this script (exact and reliable)
    for reco, rel in HIT_COLLECTIONS.items():
        if rel == rel_name:
            out["from"].append(reco)
            out["to"].append(SIM_COLLECTIONS[reco])

    # Calo-like pattern: <Prefix>RelationsSim<Suffix>
    if "RelationsSim" in rel_name:
        prefix, suffix = rel_name.split("RelationsSim", 1)
        out["from"].extend([
            f"{prefix}Collection{suffix}",
            f"{prefix}Hits{suffix}",
            f"{prefix}{suffix}",
        ])
        out["to"].extend([
            f"{prefix}Collection",
            f"{prefix}Hits",
            prefix,
        ])

    # Remove duplicates, keep order
    for k in ("from", "to"):
        seen = set()
        uniq = []
        for c in out[k]:
            if c and c not in seen:
                uniq.append(c)
                seen.add(c)
        out[k] = uniq
    return out


def _infer_relation_connections(chain, branch_names):
    """Infer relation collection endpoints as names when possible, else by collection ID."""
    relation_names = []
    for b in sorted(branch_names):
        from_name = f"_{b}_from"
        to_name = f"_{b}_to"
        if not (_has_branch(chain, from_name) and _has_branch(chain, to_name)):
            continue
        # Some branches exist in the tree but are not python-readable as attributes.
        # Keep only relation collections whose endpoints we can actually access.
        if _get_branch_obj(chain, from_name) is None or _get_branch_obj(chain, to_name) is None:
            continue
        relation_names.append(b)

    # Map collectionID -> name from heuristic anchors on '_from' / '_to' endpoints
    id_to_name = {}
    for rel_name in relation_names:
        rel_from = _get_branch_obj(chain, f"_{rel_name}_from")
        rel_to = _get_branch_obj(chain, f"_{rel_name}_to")
        if rel_from is None or rel_to is None:
            continue
        if len(rel_from) == 0 and len(rel_to) == 0:
            continue

        cands = _relation_candidates(rel_name)
        from_name = next((_find_branch_ci(c, branch_names) for c in cands["from"] if _find_branch_ci(c, branch_names)), None)
        to_name = next((_find_branch_ci(c, branch_names) for c in cands["to"] if _find_branch_ci(c, branch_names)), None)

        if len(rel_from) > 0 and from_name is not None:
            cid = int(_safe_getattr(rel_from[0], ["collectionID"], default=-1))
            if cid >= 0:
                id_to_name[cid] = from_name
        if len(rel_to) > 0 and to_name is not None:
            cid = int(_safe_getattr(rel_to[0], ["collectionID"], default=-1))
            if cid >= 0:
                id_to_name[cid] = to_name

    connections = []
    for rel_name in relation_names:
        rel_from = _get_branch_obj(chain, f"_{rel_name}_from")
        rel_to = _get_branch_obj(chain, f"_{rel_name}_to")
        if rel_from is None or rel_to is None:
            continue

        from_cid = int(_safe_getattr(rel_from[0], ["collectionID"], default=-1)) if len(rel_from) > 0 else -1
        to_cid = int(_safe_getattr(rel_to[0], ["collectionID"], default=-1)) if len(rel_to) > 0 else -1

        cands = _relation_candidates(rel_name)
        from_name = id_to_name.get(from_cid)
        to_name = id_to_name.get(to_cid)

        if from_name is None:
            from_name = next((_find_branch_ci(c, branch_names) for c in cands["from"] if _find_branch_ci(c, branch_names)), None)
        if to_name is None:
            to_name = next((_find_branch_ci(c, branch_names) for c in cands["to"] if _find_branch_ci(c, branch_names)), None)

        from_txt = from_name if from_name is not None else (f"unknown (collectionID={from_cid})" if from_cid >= 0 else "unknown")
        to_txt = to_name if to_name is not None else (f"unknown (collectionID={to_cid})" if to_cid >= 0 else "unknown")
        connections.append((rel_name, from_txt, to_txt, len(rel_from)))

    return connections


def collect_hits(input_file, max_events=1, collections=None, surfaces_csv=None, debug=False):
    """Return ({subdet: {category: features}}, {subdet: {category: features}})."""
    collections = collections or list(HIT_COLLECTIONS)
    data_hits = {
        sd: {c: {f: [] for f in FEATURES} for c in CATEGORIES}
        for sd in collections
    }
    data_pixels = {
        sd: {c: {f: [] for f in FEATURES} for c in CATEGORIES}
        for sd in collections
    }
    surfaces = _load_surfaces_map(surfaces_csv)
    if debug:
        print(f"[debug] surfaces loaded: {len(surfaces)} entries")

    chain = ROOT.TChain("events")
    added = chain.Add(input_file)
    if added == 0:
        raise RuntimeError(f"Could not open input file or tree 'events': {input_file}")

    available = [c for c in collections if _has_branch(chain, c)]
    missing = [c for c in collections if c not in available]
    if missing:
        print("Missing tracker-hit collections in input (skipping):", ", ".join(missing))
    if not available:
        raise RuntimeError("None of the requested tracker-hit collections are available in tree 'events'.")

    n_entries = chain.GetEntries()
    if max_events is None or max_events <= 0:
        n_events = n_entries
    else:
        n_events = min(n_entries, max_events)

    for ievt in range(n_events):
        chain.GetEntry(ievt)
        mc_particles = getattr(chain, "MCParticles") if _has_branch(chain, "MCParticles") else []

        for col in available:
            rel_col = HIT_COLLECTIONS[col]
            sim_col = SIM_COLLECTIONS[col]
            pix_col = PIXEL_COLLECTIONS[col]
            raw_rel_col = RAW_REL_COLLECTIONS[col]
            subdet = SURFACE_SUBDETS[col]

            hits = getattr(chain, col)
            rel_from = getattr(chain, f"_{rel_col}_from") if _has_branch(chain, f"_{rel_col}_from") else None
            rel_to = getattr(chain, f"_{rel_col}_to") if _has_branch(chain, f"_{rel_col}_to") else None
            sim_to_mc = getattr(chain, f"_{sim_col}_particle") if _has_branch(chain, f"_{sim_col}_particle") else None
            sim_hits = _get_branch_obj(chain, sim_col)

            pixels = _get_branch_obj(chain, pix_col)
            raw_from = _get_branch_obj(chain, f"_{raw_rel_col}_from")
            raw_to = _get_branch_obj(chain, f"_{raw_rel_col}_to")

            # For tracker HitsRelations in EDM4hep: from = reco hit, to = sim hit.
            # Keep this explicit mapping to avoid accidental mis-orientation.
            rel_map = _relation_map(rel_from, rel_to)
            hit_to_pixels = {}
            if pixels is not None:
                hit_to_pixels = _build_hit_to_pixel_map(raw_from, raw_to, len(hits), len(pixels))

            debug_col_counts = {k: 0 for k in CATEGORIES}
            debug_simidx_valid = {k: 0 for k in CATEGORIES}
            debug_basis_found = {k: 0 for k in CATEGORIES}
            debug_geom_filled = {k: 0 for k in CATEGORIES}
            debug_missing_cellids = []

            for ihit, hit in enumerate(hits):
                sim_index = rel_map.get(ihit)
                cat = _classify_hit(sim_index, sim_to_mc, mc_particles)
                debug_col_counts[cat] += 1

                x, y, z = _get_xyz(hit)
                r = float(np.hypot(x, y))

                d = data_hits[col][cat]
                d["x"].append(x); d["y"].append(y); d["z"].append(z)
                d["r"].append(r)
                d["phi"].append(float(np.arctan2(y, x)))
                d["theta"].append(float(np.arctan2(r, z)))
                d["t"].append(float(_safe_getattr(hit, ["time", "getTime"], default=np.nan)))
                d["edep"].append(float(_safe_getattr(hit, ["eDep", "EDep", "getEDep"], default=np.nan)))
                d["du"].append(float(_safe_getattr(hit, ["du", "getDu"], default=np.nan)))
                d["dv"].append(float(_safe_getattr(hit, ["dv", "getDv"], default=np.nan)))

                # Cluster variables from fired pixels linked to this hit.
                pix_indices = hit_to_pixels.get(ihit, [])
                d["npix"].append(float(len(pix_indices)))

                if pix_indices:
                    us, vs, charges, times = [], [], [], []
                    for ipix in pix_indices:
                        pix = pixels[ipix]
                        px, py, pz = _get_xyz(pix)
                        pr = float(np.hypot(px, py))

                        # Associated pixel features (same category as parent hit).
                        dp = data_pixels[col][cat]
                        dp["x"].append(px); dp["y"].append(py); dp["z"].append(pz)
                        dp["r"].append(pr)
                        dp["phi"].append(float(np.arctan2(py, px)))
                        dp["theta"].append(float(np.arctan2(pr, pz)))
                        pt = float(_safe_getattr(pix, ["time", "getTime"], default=np.nan))
                        dp["t"].append(pt)
                        q = float(_safe_getattr(pix, ["eDep", "EDep", "getEDep"], default=np.nan))
                        dp["edep"].append(q)
                        dp["du"].append(float(_safe_getattr(pix, ["du", "getDu"], default=np.nan)))
                        dp["dv"].append(float(_safe_getattr(pix, ["dv", "getDv"], default=np.nan)))
                        dp["pix_charge"].append(q)

                        us.append(px)
                        vs.append(py)
                        charges.append(q)
                        times.append(pt)

                    ext_u = float(int(round(max(us) - min(us))) + 1 if us else 0)
                    ext_v = float(int(round(max(vs) - min(vs))) + 1 if vs else 0)
                    d["ext_u"].append(ext_u)
                    d["ext_v"].append(ext_v)
                    for q in charges:
                        d["pix_charge"].append(q)

                    q_arr = np.asarray(charges, dtype=float)
                    t_arr = np.asarray(times, dtype=float)
                    npix = float(len(q_arr))
                    q_sum = float(np.nansum(q_arr))
                    q_max = float(np.nanmax(q_arr)) if q_arr.size else np.nan
                    q_mean = float(np.nanmean(q_arr)) if q_arr.size else np.nan
                    q_std = float(np.nanstd(q_arr)) if q_arr.size else np.nan
                    q_peak_frac = float(q_max / q_sum) if (np.isfinite(q_max) and np.isfinite(q_sum) and q_sum > 0) else np.nan

                    if np.isfinite(q_sum) and q_sum > 0:
                        w = q_arr / q_sum
                        w = w[np.isfinite(w) & (w > 0)]
                        q_entropy = float(-np.sum(w * np.log(w))) if w.size else np.nan
                    else:
                        q_entropy = np.nan

                    fill_factor = float(npix / (ext_u * ext_v)) if ext_u > 0 and ext_v > 0 else np.nan
                    ext_ratio = float(ext_v / ext_u) if ext_u > 0 else np.nan
                    t_span_pix = float(np.nanmax(t_arr) - np.nanmin(t_arr)) if t_arr.size else np.nan
                    t_std_pix = float(np.nanstd(t_arr)) if t_arr.size else np.nan

                    if len(us) >= 2:
                        coords = np.column_stack((np.asarray(us, dtype=float), np.asarray(vs, dtype=float)))
                        cov = np.cov(coords, rowvar=False)
                        eig = np.linalg.eigvalsh(cov)
                        major = float(max(eig[-1], 0.0))
                        minor = float(max(eig[0], 0.0))
                        pca_axis_ratio = float(np.sqrt(major / minor)) if minor > 0 else np.nan
                    else:
                        pca_axis_ratio = np.nan

                    d["q_sum"].append(q_sum)
                    d["q_max"].append(q_max)
                    d["q_mean"].append(q_mean)
                    d["q_std"].append(q_std)
                    d["q_peak_frac"].append(q_peak_frac)
                    d["q_entropy"].append(q_entropy)
                    d["fill_factor"].append(fill_factor)
                    d["ext_ratio"].append(ext_ratio)
                    d["t_span_pix"].append(t_span_pix)
                    d["t_std_pix"].append(t_std_pix)
                    d["pca_axis_ratio"].append(pca_axis_ratio)
                else:
                    d["ext_u"].append(0.0)
                    d["ext_v"].append(0.0)
                    d["q_sum"].append(np.nan)
                    d["q_max"].append(np.nan)
                    d["q_mean"].append(np.nan)
                    d["q_std"].append(np.nan)
                    d["q_peak_frac"].append(np.nan)
                    d["q_entropy"].append(np.nan)
                    d["fill_factor"].append(np.nan)
                    d["ext_ratio"].append(np.nan)
                    d["t_span_pix"].append(np.nan)
                    d["t_std_pix"].append(np.nan)
                    d["pca_axis_ratio"].append(np.nan)

                # Geometry-based variables (need sim hit + surface axes U,V,N).
                tan_u = np.nan
                tan_v = np.nan
                res_v = np.nan
                pull_v = np.nan
                if (
                    sim_index is not None
                    and sim_hits is not None
                    and 0 <= sim_index < len(sim_hits)
                ):
                    debug_simidx_valid[cat] += 1
                    sim_hit = sim_hits[sim_index]
                    cellid = _get_cellid(hit)
                    basis = surfaces.get((subdet, cellid))
                    if basis is not None:
                        debug_basis_found[cat] += 1
                        U, V, N = basis
                        spx, spy, spz = _get_xyz(sim_hit)
                        pmx, pmy, pmz = _get_momentum_xyz(sim_hit)
                        p = np.array([pmx, pmy, pmz], dtype=float)
                        pn = float(np.dot(p, N))
                        if np.isfinite(pn) and abs(pn) > 0:
                            tan_u = float(np.dot(p, U) / pn)
                            tan_v = float(np.dot(p, V) / pn)
                        dpos = np.array([x - spx, y - spy, z - spz], dtype=float)
                        res_v = float(1e3 * np.dot(dpos, V))  # um
                        dv_mm = float(_safe_getattr(hit, ["dv", "getDv"], default=np.nan))
                        if np.isfinite(dv_mm) and abs(dv_mm) > 0:
                            pull_v = float(res_v / (1e3 * dv_mm))
                        if np.isfinite(tan_u) or np.isfinite(tan_v) or np.isfinite(res_v) or np.isfinite(pull_v):
                            debug_geom_filled[cat] += 1
                    elif debug and len(debug_missing_cellids) < 5:
                        debug_missing_cellids.append(cellid)

                d["tan_u"].append(tan_u)
                d["tan_v"].append(tan_v)
                d["res_v"].append(res_v)
                d["pull_v"].append(pull_v)

            if debug:
                n_rel = min(len(rel_from), len(rel_to)) if (rel_from is not None and rel_to is not None) else 0
                n_raw = min(len(raw_from), len(raw_to)) if (raw_from is not None and raw_to is not None) else 0
                n_sim = len(sim_hits) if sim_hits is not None else 0
                n_pix = len(pixels) if pixels is not None else 0
                n_h2p = sum(len(v) for v in hit_to_pixels.values())
                print(f"[debug] evt={ievt} col={col}: hits={len(hits)} rel={n_rel} simhits={n_sim} pixels={n_pix} rawRel={n_raw} mappedHitPixelLinks={n_h2p}")
                print(f"[debug] evt={ievt} col={col}: category counts {debug_col_counts}")
                print(f"[debug] evt={ievt} col={col}: valid sim-index by category {debug_simidx_valid}")
                print(f"[debug] evt={ievt} col={col}: surface basis found by category {debug_basis_found}")
                print(f"[debug] evt={ievt} col={col}: geometry vars filled by category {debug_geom_filled}")
                if debug_missing_cellids:
                    print(f"[debug] evt={ievt} col={col}: example missing cellIDs (first 5): {debug_missing_cellids}")

        print(f"Processed event {ievt}")

    out_hits = {
        sd: {c: {f: np.asarray(v) for f, v in feats.items()} for c, feats in by_cat.items()}
        for sd, by_cat in data_hits.items()
    }
    out_pixels = {
        sd: {c: {f: np.asarray(v) for f, v in feats.items()} for c, feats in by_cat.items()}
        for sd, by_cat in data_pixels.items()
    }
    return out_hits, out_pixels


def plot_hit_features(data, features=FEATURES, bins=60, output_dir=None, file_prefix="hit_feature", title_prefix=""):
    subdets = [sd for sd in SUBDETECTOR_ORDER if sd in data]
    total_counts = {
        cat: int(sum(len(data[sd][cat]["x"]) for sd in subdets))
        for cat in CATEGORIES
    }
    print(f"{title_prefix}counts:" if title_prefix else "Hit counts:", total_counts)
    for sd in subdets:
        per_sd = {cat: int(len(data[sd][cat]["x"])) for cat in CATEGORIES}
        print(f"  {sd}: {per_sd}")

    for feature in features:
        if feature in SIMHIT_REQUIRED_FEATURES:
            finite_by_cat = {}
            for cat in CATEGORIES:
                parts = [data[sd][cat][feature] for sd in subdets if len(data[sd][cat][feature])]
                finite_by_cat[cat] = int(sum(np.isfinite(p).sum() for p in parts)) if parts else 0
            if any(v == 0 for v in finite_by_cat.values()):
                print(f"[info] feature '{feature}' requires valid SimTrackerHit + surface axes; finite entries by category: {finite_by_cat}")

        # Common binning across all categories
        arrays = [
            data[sd][cat][feature]
            for sd in subdets
            for cat in CATEGORIES
            if len(data[sd][cat][feature])
        ]
        if not arrays:
            continue

        allv = np.concatenate(arrays)
        allv = allv[np.isfinite(allv)]
        if allv.size == 0:
            continue

        if feature in INT_FEATURES:
            iv = np.rint(allv).astype(int)
            lo_i = int(iv.min())
            hi_i = int(iv.max())
            # Unit-width bins centered on integer values: [..., n-0.5, n+0.5, ...]
            edges = np.arange(lo_i - 0.5, hi_i + 1.5, 1.0)
        else:
            lo, hi = np.percentile(allv, [0.1, 99.9])
            if feature in LOGX:
                positive = allv[allv > 0]
                if positive.size == 0:
                    continue
                lo = max(lo, positive.min())
                hi = max(hi, lo * 1.001)
                edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
            else:
                edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins

        if feature in LOGX:
            positive = allv[allv > 0]
            if positive.size == 0:
                continue

        fig, axes = plt.subplots(len(subdets), 2, figsize=(15, 2.8 * len(subdets)), sharex=True)
        if len(subdets) == 1:
            axes = np.array([axes])
        suptitle = f"{title_prefix} {LABELS.get(feature, feature)}".strip()
        fig.suptitle(suptitle)

        for i, sd in enumerate(subdets):
            for j, density in enumerate((False, True)):
                ax = axes[i, j]
                any_drawn = False
                for cat, (label, color) in CATEGORIES.items():
                    vals = data[sd][cat][feature]
                    if len(vals) == 0:
                        continue
                    vals = vals[np.isfinite(vals)]
                    if feature in LOGX:
                        vals = vals[vals > 0]
                    if len(vals) == 0:
                        continue

                    any_drawn = True
                    ax.hist(vals, bins=edges, histtype="step", linewidth=2.0,
                            color=color, density=density, label=f"{label} (N={len(vals)})")

                ax.set_ylabel((f"{sd}\n" + ("Normalized entries" if density else "Entries")) if j == 0 else ("Normalized entries" if density else "Entries"))
                ax.set_yscale("log")
                if feature in LOGX:
                    ax.set_xscale("log")
                ax.grid(which="major", alpha=0.28)
                ax.grid(which="minor", alpha=0.12)
                ax.minorticks_on()
                if feature in INT_FEATURES:
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                if i == 0:
                    ax.set_title("Normalized" if density else "Absolute")
                if i == len(subdets) - 1:
                    ax.set_xlabel(LABELS.get(feature, feature))
                if any_drawn:
                    ax.legend(loc="best", frameon=True, framealpha=0.9)
                else:
                    ax.text(0.5, 0.5, "No entries", transform=ax.transAxes,
                            ha="center", va="center", color="gray", fontsize=10)

        fig.tight_layout(rect=(0, 0, 1, 0.96))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f"{file_prefix}_{feature}.png"), dpi=150)
            plt.close(fig)
        else:
            plt.show()


def print_avail_features(input_file):
    chain = ROOT.TChain("events")
    if chain.Add(input_file) == 0:
        raise RuntimeError(f"Could not open input file or tree 'events': {input_file}")

    if chain.GetEntries() == 0:
        print("No events in tree 'events'.")
        return

    chain.GetEntry(0)

    print("All branches/collections in tree 'events':")
    branch_names = sorted(b.GetName() for b in chain.GetListOfBranches())
    for name in branch_names:
        if name.startswith("podio"):
            continue
        #if not (name.startswith("V") and "Barrel" in name):
        #    continue
        try:
            obj = getattr(chain, name)
            size = len(obj)
            print(f"  {name}: {size} entries")
        except Exception:
            print(f"  {name}")
    print()

    relations = _infer_relation_connections(chain, branch_names)
    if relations:
        print("Relation collections and connected collections:")
        for rel_name, from_txt, to_txt, nlinks in relations:
            print(f"  {rel_name}: {from_txt} -> {to_txt} (N={nlinks})")
        print()

    print("Available tracker-hit collections in first event:")
    for col in HIT_COLLECTIONS:
        if _has_branch(chain, col):
            try:
                hits = getattr(chain, col)
                print(f"  {col}: {len(hits)} hits")
            except Exception:
                print(f"  {col}: branch exists, but could not read")
        else:
            print(f"  {col}: not available")

    
    for col in list(HIT_COLLECTIONS) + list(PIXEL_COLLECTIONS.values()) + ["VertexBarrelCollection"]:
        print(f"collection '{col}':")
        if not _has_branch(chain, col):
            print("  not available")
            continue
        obj = _get_branch_obj(chain, col)
        if obj is None or len(obj) == 0:
            print("  empty or unreadable")
            continue
        print([a for a in dir(obj[0]) if not a.startswith("_")])


def test_uv_alignment(input_file, surfaces_csv, samples_per_subdet=5):
    """Quick sanity check of local surface axes using a few hits from the first event.

    Prints |u·z|, |v·z| and |u·phi|, |v·phi| where phi is the local azimuthal
    unit vector at the hit position in the transverse plane.
    """
    surfaces = _load_surfaces_map(surfaces_csv)
    if not surfaces:
        raise RuntimeError("UV alignment test needs --surfaces-csv with valid content")

    chain = ROOT.TChain("events")
    if chain.Add(input_file) == 0:
        raise RuntimeError(f"Could not open input file or tree 'events': {input_file}")
    if chain.GetEntries() == 0:
        print("No events in tree 'events'.")
        return

    chain.GetEntry(0)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)

    print("UV alignment test on first event")
    print(f"surfaces loaded: {len(surfaces)}")
    print(f"samples per subdetector: {samples_per_subdet}")

    for col in SUBDETECTOR_ORDER:
        if not _has_branch(chain, col):
            continue
        hits = _get_branch_obj(chain, col)
        if hits is None or len(hits) == 0:
            continue

        subdet = SURFACE_SUBDETS[col]
        n_printed = 0
        print(f"\n[{col}]")
        for ihit, hit in enumerate(hits):
            x, y, z = _get_xyz(hit)
            rt = float(np.hypot(x, y))
            if not np.isfinite(rt) or rt <= 0:
                continue

            cellid = _get_cellid(hit)
            basis = surfaces.get((subdet, cellid))
            if basis is None:
                continue
            U, V, _N = basis

            # local azimuthal unit vector at hit position
            ephi = np.array([-y / rt, x / rt, 0.0], dtype=float)

            uz = abs(float(np.dot(U, ez)))
            vz = abs(float(np.dot(V, ez)))
            uphi = abs(float(np.dot(U, ephi)))
            vphi = abs(float(np.dot(V, ephi)))

            print(
                f"  hit {ihit:5d}: |u·z|={uz:7.4f}  |v·z|={vz:7.4f}  "
                f"|u·phi|={uphi:7.4f}  |v·phi|={vphi:7.4f}"
            )
            n_printed += 1
            if n_printed >= samples_per_subdet:
                break

        if n_printed == 0:
            print("  no printable hits with matching surface entry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tracker hit features for signal, sim-particle, BIB and noise hits (ROOT-only, no podio)")
    parser.add_argument("input_file", nargs="?",
                        default="/data/dust/user/fmeloni/DataMuC/digiBIB/multiMuonGun_realistic_001BIB/multiMuonGun_digi_0.edm4hep.root")
    parser.add_argument(
        "-n", "--nevents", "--max-events",
        dest="max_events",
        type=int,
        default=1,
        help="Number of events to process (use negative value to process all events)",
    )
    parser.add_argument("-o", "--output-dir", default="hit_feature_plots")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--features", nargs="+", default=FEATURES, choices=FEATURES)
    parser.add_argument("--surfaces-csv", default="/data/dust/user/bauckhag/code/TrackML/source/data_generation/EDM4hep_extractor/surfaces/surfaces.csv",
                        help="Optional surfaces.csv with columns subdet,cellID,ux,uy,uz,vx,vy,vz,nx,ny,nz "
                             "(needed for tan_u/tan_v/res_v/pull_v)")
    parser.add_argument("--test-uv-align", action="store_true",
                        help="run a quick UV axis alignment test on the first event")
    parser.add_argument("--test-samples", type=int, default=5,
                        help="number of hit samples per subdetector for --test-uv-align")
    parser.add_argument("--debug", action="store_true", help="print debug counters for relation/surface matching")
    parser.add_argument("--list", action="store_true", help="print available hit attributes and exit")
    args = parser.parse_args()

    _setup_plot_style()

    if args.list:
        print_avail_features(args.input_file)
    elif args.test_uv_align:
        test_uv_alignment(args.input_file, args.surfaces_csv, samples_per_subdet=max(1, args.test_samples))
    else:
        hit_data, pixel_data = collect_hits(
            args.input_file,
            max_events=args.max_events,
            surfaces_csv=args.surfaces_csv,
            debug=args.debug,
        )
        plot_hit_features(
            hit_data,
            args.features,
            bins=args.bins,
            output_dir=args.output_dir,
            file_prefix="hit_feature",
            title_prefix="Hit",
        )
        plot_hit_features(
            pixel_data,
            args.features,
            bins=args.bins,
            output_dir=args.output_dir,
            file_prefix="pixel_feature",
            title_prefix="Associated pixel",
        )
