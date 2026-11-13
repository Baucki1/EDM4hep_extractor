# EDM4hep extractor

Converts reconstructed tracker hits from EDM4hep (`.edm4hep.root`) files of the MuColl sim/reco chain into per-event CSV files for the acorn ML pipeline. Each reco hit is matched to its simulated hit, so every output row carries truth information (particle, true position and momentum) alongside the detector-level measurement.

## Contents

| File(s) | Purpose |
| --- | --- |
| `extract_edm4hep.py` | Command-line entry point. Parse options and runs the extractor. |
| `extractor_edm4hep.py` | `edm4hep_extractor` class: reads events, builds the CSV tables, optional seeds and plots. |
| `hit_features.py` | Per-hit geometric features (incidence angles, corrected time, pixel-cluster residuals). |
| `seed_utils.py` | Helix seeds from truth tracklets (true, duplicate and fake) and helix-to-hit residuals. |
| `plot_hit_features.py`, `plot_hit_features_root.py` | Plots of the per-hit features (podio and ROOT version). |
| `surfaces/` | [`export_surfaces.py`](https://github.com/MuonColliderSoft/MuonCVXDDigitiser/blob/master/test/validation/export_surfaces.py) writes the sensor surfaces (`surfaces.csv`) from the detector XML. [`common.py`](https://github.com/MuonColliderSoft/MuonCVXDDigitiser/blob/master/test/validation/common.py) holds shared definitions. Taken from [MuonColliderSoft/MuonCVXDigitiser](https://github.com/MuonColliderSoft/MuonCVXDDigitiser) |
| `extract.sh`, `extract.submit` | Wrapper and HTCondor submit file for running the extractor on the batch farm through Apptainer. |
| `test_*.py` | Unit tests for the feature, seed and seed-writer code. |

## Requirements

- The MuColl software stack, which provides `podio` and `edm4hep`.
- Python with `numpy`, `pandas` and `matplotlib`.
- `pytest` for the tests.

## Usage

```bash
python3 extract_edm4hep.py \
    --input  /path/to/edm4hep_files \
    --output /path/to/csv_output \
    --n_events 100 --n_workers 1 --i_worker 0 \
    --surfaces surfaces/surfaces.csv --seeds
```

The input directory is scanned for files ending in `.root`, sorted by name, and read as one event stream.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--input` | | Directory with the EDM4hep files. |
| `--output` | | Output directory, created if missing. |
| `--n_events` | `1000` | Number of events to process in total. |
| `--n_workers`, `--i_worker` | `1`, `0` | Split the events into `n_workers` contiguous ranges and process range `i_worker` (easy to parallelize with condor/slurm). |
| `--timing` | `True` | Include the reco hit time `t`, the simulated hit time `tt` and the particle vertex time `vt`. |
| `--sim_particles_to_bkg` | `False` | Map particles created in the simulation to background (id 0). |
| `--noise_to_bkg` | `True` | Map noise hits to background particle (id 0). |
| `--coned_hits` | `False` | Accepted for interface compatibility with the LCIO extractor. It has no effect here. |
| `--pixel_cluster_features` | `True` | Add per-cluster pixel features (`npix`, `ext_u`, `ext_v`) and, with `--surfaces`, the expected-extent residuals. |
| `--surfaces` | `None` | `surfaces.csv` from `surfaces/export_surfaces.py`. Enables the IP-pointing and time-of-flight hit features. |
| `--seeds` | off | Also write `eventN-seeds.csv` with truth-tracklet helix seeds. |
| `--seed_min_purity` | `1.0` | Fraction of a seed's three hits that must share a particle for the seed to count as true. |
| `--seed_duplicates` | `1` | Extra tracklets per particle, labelled as duplicates. |
| `--seed_fake_ratio` | `5.0` | Fake seeds per primary seed. |
| `--seed_max_start` | `3` | Tracklets start within the first N hit triplets of a particle. |
| `--seed_random_seed` | `0` | Base seed. Each event derives its own random generator from it and the event number. |

> **Note:** `--timing`, `--sim_particles_to_bkg`, `--noise_to_bkg`, `--coned_hits` and `--pixel_cluster_features` are declared with `type=bool`. Argparse converts any non-empty string to `True`, so `--timing False` still means `True`. Leave these at their defaults or edit the default in the script instead.

## Input collections

The extractor reads the reconstructed tracker hits and their links to simulated hits. The collection names follow the MuColl reconstruction:

| Reco hits | Sim hit relation | Fired pixels | Subdetector |
| --- | --- | --- | --- |
| `VXDBarrelHits` | `VXDBarrelHitsRelations` | `VXDBarrelPixels` | `VXDBarrel` |
| `ITBarrelHits` | `ITBarrelHitsRelations` | | `ITBarrel` |
| `OTBarrelHits` | `OTBarrelHitsRelations` | | `OTBarrel` |
| `VXDEndcapHits` | `VXDEndcapHitsRelations` | `VXDEndcapPixels` | `VXDEndcap` |
| `ITEndcapHits` | `ITEndcapHitsRelations` | | `ITEndcap` |
| `OTEndcapHits` | `OTEndcapHitsRelations` | | `OTEndcap` |

Hits without a related simulated hit are skipped. The `cellID` of each hit is decoded with the encoding string stored in the metadata frame (`<SimCollection>__CellIDEncoding`), giving `volume_id`, `layer_id`, `module_id` and `sensor_id`. The raw hit-to-pixel relations (`*RawHitRelations`) provide the pixel-cluster features when `--pixel_cluster_features` is on.

## Output

For each event `N` (zero-padded to nine digits) the extractor writes:

| File | Columns |
| --- | --- |
| `eventN-hits.csv` | `hit_id, x, y, z, [t,] volume_id, layer_id, module_id`, plus the geometric feature columns below (only with `--pixel_cluster_features`). |
| `eventN-truth.csv` | `hit_id, particle_id, tx, ty, tz, [tt,] tpx, tpy, tpz, weight` |
| `eventN-cells.csv` | `hit_id, ch0, ch1, value` (`ch0` is the cell ID; `value` is the deposited energy). |
| `eventN-particles.csv` | `particle_id, vx, vy, vz, [vt,] px, py, pz, q, nhits` |
| `eventN-seeds.csv` | Only with `--seeds`. See below. |

Lengths are in mm, times in ns and momenta in GeV (EDM4hep units). `[t]`, `[tt]` and `[vt]` are present when `--timing` is on.

The hit files are written from the same table, each with its own column subset. Rows in `hits`, `truth` and `cells` share `hit_id`, so they can be joined.

### Hit features

With `--surfaces`, the hit file gets geometric features computed from the sensor surfaces:

- `inc_cos`: |cos| of the incidence angle of a straight track from the origin through the hit.
- `tan_u_ip`, `tan_v_ip`: tangent of the IP-pointing angle along the sensor's u and v axes.
- `t_corr`: hit time minus the time of flight from the origin (with `--timing`).
- `ext_u_exp`, `ext_v_exp`, `ext_u_res`, `ext_v_res`: expected cluster extent for an IP-pointing track and its residual against the measured extent (with `--pixel_cluster_features`).

Features are NaN for sensors missing from `surfaces.csv`; the extractor logs a warning with the count.

### Seeds

With `--seeds`, `eventN-seeds.csv` contains helix seeds built from three consecutive truth hits per tracklet.

**Seed building** 
For each particle with at least three hits, its hits are sorted by distance from the origin. A start position is picked at random among the first `--seed_max_start` triplets, and the three consecutive hits from there form the primary tracklet. 
- `--seed_duplicates` further tracklets are drawn the same way from other start positions of that particle. They are labelled as duplicates (`origin` = 1) and may share hits with the primary. Particle 0 (background and noise) never seeds a tracklet. 
- Each event uses its own random generator, seeded from `--seed_random_seed` and the event number, so the output is reproducible.
- A seed is `true` when its majority particle covers at least `--seed_min_purity` of its three hits (default: all three).
- Fake seeds are drawn from nearby outward hits, so fewer than `fake_ratio` per primary may appear in sparse regions.

**Helix fit.** Each triplet is fitted in three steps:

1. In the xy plane, the circle through the three points gives the signed curvature `kappa = 1/R`. Based on the curvature one of two hypotheses is used: If `|kappa|` is below `1e-9`, the triplet is treated as a straight line.
2. The circle gives the perigee point and direction, from which `phi0` and `d0` follow. Each hit's transverse arc length `s` from the perigee is the angle it subtends on the circle, times `R`.
3. In the (s, z) plane, a least-squares line gives `tan_lambda` and `z0`. The time `t0` is the mean over the hits of `t - s*sqrt(1 + tan_lambda^2)/c`, and is NaN without hit times.

 Seeds whose first five parameters (`d0, z0, phi0, tan_lambda, kappa`) are not finite are dropped. Each row has:

- `seed_id`, `hit_id_0`, `hit_id_1`, `hit_id_2` (ordered inside out), and `particle_id` (majority particle; 0 if none).
- Helix parameters `d0, z0, phi0, tan_lambda, kappa, t0` (perigee with respect to the origin; `kappa = 1/R` is signed; `t0` is NaN without timing).
- `purity`, `is_true`, and `origin` (0 = true primary, 1 = duplicate, 2 = fake).



## Surfaces

`surfaces/export_surfaces.py` writes `surfaces.csv` (cellID to origin, u, v and normal vectors, lengths in mm) from the MAIA_v0 detector description. It needs DD4hep and DDG4 from the MuColl stack:

```bash
python3 surfaces/export_surfaces.py <MAIA_v0.xml> surfaces/surfaces.csv
```

## Plotting

`plot_hit_features.py` (podio) and `plot_hit_features_root.py` (ROOT) make per-feature histograms from the EDM4hep files.

## Tests

The unit tests cover the hit features, the seed generator and the seed writer. They do not need `podio` or `edm4hep`:

```bash
/data/dust/user/bauckhag/code/TrackML/venv/mamba_acorn/bin/python -m pytest -q \
    test_hit_features.py test_seed_utils.py test_extractor_seeds.py
```