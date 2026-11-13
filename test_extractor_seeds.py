import sys
import types

import numpy as np
import pandas as pd

# podio is only available in the simulation image; the seed writer does not need it.
podio = types.ModuleType("podio")
podio.root_io = types.ModuleType("podio.root_io")
sys.modules.setdefault("podio", podio)
sys.modules.setdefault("podio.root_io", podio.root_io)
sys.modules.setdefault("edm4hep", types.ModuleType("edm4hep"))

import extractor_edm4hep as ex
import test_seed_utils as tsu


def make_extractor(tmp_path, **seed_options):
    extractor = object.__new__(ex.edm4hep_extractor)
    extractor.output_dir = str(tmp_path)
    extractor.timing = True
    extractor.seeds = True
    extractor.seed_options = dict(min_purity=1.0, n_duplicates=1, fake_ratio=3.0, max_start=3)
    extractor.seed_options.update(seed_options)
    extractor.seed_random_seed = 5
    return extractor


def make_hits():
    _, xyz, pid, t = tsu.make_event(np.random.default_rng(0))
    return pd.DataFrame(
        {"hit_id": np.arange(len(xyz), dtype=float), "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
         "t": t, "particle_id": pid.astype(float)}
    )


def test_seed_csv_refers_to_hit_ids_and_is_reproducible(tmp_path):
    hits = make_hits()
    extractor = make_extractor(tmp_path)

    extractor._write_seeds(hits, 7)
    first = pd.read_csv(tmp_path / "event000000007-seeds.csv")
    extractor._write_seeds(hits, 7)
    second = pd.read_csv(tmp_path / "event000000007-seeds.csv")

    assert first.equals(second)
    assert list(first.columns[:4]) == ["seed_id", "hit_id_0", "hit_id_1", "hit_id_2"]
    assert set(first["hit_id_0"]) <= set(hits["hit_id"].astype(int))
    assert (first["origin"] == 0).sum() == 4 and (first["origin"] == 1).sum() == 4
    assert first.loc[first["origin"] < 2, "is_true"].all()


def test_min_purity_option_changes_the_truth_flag(tmp_path):
    hits = make_hits()
    strict, loose = tmp_path / "strict", tmp_path / "loose"
    strict.mkdir(), loose.mkdir()

    make_extractor(strict, fake_ratio=30.0)._write_seeds(hits, 1)
    make_extractor(loose, fake_ratio=30.0, min_purity=0.6)._write_seeds(hits, 1)

    assert (pd.read_csv(loose / "event000000001-seeds.csv")["is_true"].sum()
            > pd.read_csv(strict / "event000000001-seeds.csv")["is_true"].sum())
