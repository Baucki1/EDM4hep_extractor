import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from podio import root_io

# Reco tracker-hit collections and their hit<->sim-hit link collections
HIT_COLLECTIONS = {
    "VXDBarrelHits": "VXDBarrelHitsRelations",
    "ITBarrelHits": "ITBarrelHitsRelations",
    "OTBarrelHits": "OTBarrelHitsRelations",
    "VXDEndcapHits": "VXDEndcapHitsRelations",
    "ITEndcapHits": "ITEndcapHitsRelations",
    "OTEndcapHits": "OTEndcapHitsRelations",
}

# Hit categories (label, color). Classification, per reco hit:
#   noise      : no related SimTrackerHit (digitisation noise)
#   bib        : related SimTrackerHit without MCParticle (overlaid BIB)
#   sim        : MCParticle created in simulation (secondaries, mapped to background)
#   signal     : MCParticle from the generator (not created in simulation)
CATEGORIES = {
    "signal": ("Signal hits", "tab:blue"),
    "sim": ("Sim-particle hits (secondaries)", "tab:green"),
    "bib": ("BIB hits", "tab:red"),
    "noise": ("Noise hits", "tab:gray"),
}

FEATURES = ["x", "y", "z", "r", "phi", "theta", "t", "edep", "du", "dv"]
LABELS = {
    "x": "x [mm]", "y": "y [mm]", "z": "z [mm]", "r": "r [mm]",
    "phi": "phi [rad]", "theta": "theta [rad]", "t": "hit time [ns]",
    "edep": "EDep [GeV]", "du": "du [mm]", "dv": "dv [mm]",
}
# Features that benefit from a logarithmic x axis
LOGX = {"edep"}


def classify(simhit):
    if simhit is None:
        return "noise"
    particle = simhit.getParticle()
    if not particle.isAvailable():
        return "bib"
    return "sim" if particle.isCreatedInSimulation() else "signal"


def relation_map(link_collection):
    rel = {}
    for link in link_collection:
        idx = link.getFrom().getObjectID().index
        if idx not in rel:
            rel[idx] = link.getTo()
    return rel


def collect_hits(input_file, max_events=1, collections=None):
    """Return {category: {feature: np.array}} accumulated over events."""
    collections = collections or list(HIT_COLLECTIONS)
    data = {c: {f: [] for f in FEATURES} for c in CATEGORIES}

    reader = root_io.Reader(input_file)
    for ievt, event in enumerate(reader.get("events")):
        if max_events is not None and ievt >= max_events:
            break
        for col in collections:
            hits = event.get(col)
            rel = relation_map(event.get(HIT_COLLECTIONS[col]))
            for hit in hits:
                cat = classify(rel.get(hit.getObjectID().index))
                pos = hit.getPosition()
                x, y, z = pos[0], pos[1], pos[2]
                r = np.hypot(x, y)
                d = data[cat]
                d["x"].append(x); d["y"].append(y); d["z"].append(z)
                d["r"].append(r)
                d["phi"].append(np.arctan2(y, x))
                d["theta"].append(np.arctan2(r, z))
                d["t"].append(hit.getTime())
                d["edep"].append(hit.getEDep())
                d["du"].append(hit.getDu())
                d["dv"].append(hit.getDv())
        print(f"Processed event {ievt}")

    return {c: {f: np.asarray(v) for f, v in feats.items()} for c, feats in data.items()}


def plot_hit_features(data, features=FEATURES, bins=60, output_dir=None):
    counts = {c: len(d["x"]) for c, d in data.items()}
    print("Hit counts:", counts)

    for feature in features:
        # Common binning across all categories
        arrays = [d[feature] for d in data.values() if len(d[feature])]
        if not arrays:
            continue
        allv = np.concatenate(arrays)
        allv = allv[np.isfinite(allv)]
        if allv.size == 0:
            continue
        lo, hi = np.percentile(allv, [0.1, 99.9])
        if feature in LOGX:
            lo = max(lo, allv[allv > 0].min()) if (allv > 0).any() else 1e-12
            edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        else:
            edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, density in zip(axes, (False, True)):
            for cat, (label, color) in CATEGORIES.items():
                vals = data[cat][feature]
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=edges, histtype="step", linewidth=1.5,
                        color=color, density=density, label=f"{label} (N={len(vals)})")
            ax.set_xlabel(LABELS.get(feature, feature))
            ax.set_ylabel("Normalized entries" if density else "Entries")
            ax.set_yscale("log")
            if feature in LOGX:
                ax.set_xscale("log")
            ax.set_title(f"{feature} ({'normalized' if density else 'absolute'})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f"hit_feature_{feature}.png"), dpi=150)
            plt.close(fig)
        else:
            plt.show()


def print_avail_features(input_file):
    reader = root_io.Reader(input_file)
    event = reader.get("events")[0]
    for col in HIT_COLLECTIONS:
        hits = event.get(col)
        print(f"{col}: {len(hits)} hits")
    hit = event.get("VXDEndcapHits")[0]
    print("Available hit attributes:")
    print([a for a in dir(hit) if not a.startswith("_")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tracker hit features for signal, sim-particle, BIB and noise hits")
    parser.add_argument("input_file", nargs="?",
                        default="/data/dust/user/fmeloni/DataMuC/digi/multiMuonGun_realistic/multiMuonGun_reco_0.edm4hep.root")
    parser.add_argument("-n", "--max-events", type=int, default=1)
    parser.add_argument("-o", "--output-dir", default="hit_feature_plots")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--features", nargs="+", default=FEATURES, choices=FEATURES)
    parser.add_argument("--list", action="store_true", help="print available hit attributes and exit")
    args = parser.parse_args()

    if args.list:
        print_avail_features(args.input_file)
    else:
        hit_data = collect_hits(args.input_file, max_events=args.max_events)
        plot_hit_features(hit_data, args.features, bins=args.bins, output_dir=args.output_dir)
