import os, sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging

# EDM4hep / podio I/O (replaces pyLCIO)
import podio
from podio import root_io
import edm4hep

import hit_features
import seed_utils


# Collection-name mapping LCIO -> EDM4hep for this reconstruction:
#   VBTrackerHits -> VXDBarrelHits        (sim coll. VertexBarrelCollection)
#   VETrackerHits -> VXDEndcapHits        (sim coll. VertexEndcapCollection)
#   IBTrackerHits -> ITBarrelHits         (sim coll. InnerTrackerBarrelCollection)
#   IETrackerHits -> ITEndcapHits         (sim coll. InnerTrackerEndcapCollection)
#   OBTrackerHits -> OTBarrelHits         (sim coll. OuterTrackerBarrelCollection)
#   OETrackerHits -> OTEndcapHits         (sim coll. OuterTrackerEndcapCollection)
#   MCParticle    -> MCParticles
# Reco tracker hits are edm4hep::TrackerHitPlane, sim hits are edm4hep::SimTrackerHit,
# and the reco<->sim relations are podio::Link<edm4hep::TrackerHit,edm4hep::SimTrackerHit>.
# The CellID encoding string is stored in the "metadata" frame under
# "<SimCollectionName>__CellIDEncoding".


class BitFieldCoder():
    """Minimal pure-python replacement for pyLCIO UTIL.BitField64 / DD4hep BitFieldCoder.

    Parses an encoding string such as
        "system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8"
    where each field is "name:offset:width" and a negative width denotes a
    signed field. Decoding follows the same two's-complement convention used
    by DD4hep so the extracted volume/layer/module/sensor IDs match LCIO.
    """

    def __init__(self, encoding):
        self.fields = {}
        offset = 0
        for token in encoding.split(","):
            parts = token.split(":")
            name = parts[0]
            if len(parts) == 3:
                offset = int(parts[1])
                width = int(parts[2])
            else:  # "name:width" -> field placed right after the previous one
                width = int(parts[1])
            self.fields[name] = (offset, width)
            offset += abs(width)
        self._value = 0

    def setValue(self, value):
        self._value = int(value)

    def value(self, name):
        offset, width = self.fields[name]
        nbits = abs(width)
        mask = (1 << nbits) - 1
        val = (self._value >> offset) & mask
        if width < 0 and (val & (1 << (nbits - 1))):  # signed field, sign bit set
            val -= (1 << nbits)
        return val

    def value_array(self, name, values):
        """Vectorized decode of `name` over a numpy array of cellIDs."""
        offset, width = self.fields[name]
        nbits = abs(width)
        val = (values >> offset) & ((1 << nbits) - 1)
        if width < 0:  # signed field, apply two's-complement
            val = np.where(val & (1 << (nbits - 1)), val - (1 << nbits), val)
        return val


class edm4hep_extractor():
    def __init__(self, input_dir, output_dir, timing=False, sim_particles_to_bkg=True, noise_to_bkg=False,
                 coned_hits=False, pixel_cluster_features=False, surfaces=None,
                 seeds=False, seed_min_purity=1.0, seed_duplicates=1, seed_fake_ratio=5.0,
                 seed_max_start=3, seed_random_seed=0, logger=None):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.timing = timing

        self.sim_particles_to_bkg = sim_particles_to_bkg
        self.noise_to_bkg = noise_to_bkg
        self.pixel_cluster_features = pixel_cluster_features

        self.seeds = seeds
        self.seed_options = dict(
            min_purity=seed_min_purity, n_duplicates=seed_duplicates,
            fake_ratio=seed_fake_ratio, max_start=seed_max_start,
        )
        self.seed_random_seed = seed_random_seed
        # Kept for interface compatibility with the LCIO extractor. This
        # reconstruction does not provide separate "Coned" tracker-hit
        # collections, so the flag has no effect on the collection names.
        self.coned_hits = coned_hits

        if logger is None:
            logger = logging.getLogger('edm4hep_extractor')
            logger.setLevel(logging.INFO)
        self.logger = logger

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.input_files = []
        for file in os.listdir(self.input_dir):
            if file.endswith(".edm4hep.root") or file.endswith(".root"):
                self.input_files.append(os.path.join(self.input_dir, file))
        self.input_files.sort()

        # ordered list of reco tracker-hit collections (same logical order
        # VB, IB, OB, VE, IE, OE as in the LCIO extractor so the plotting
        # index conventions are preserved)
        self.tracker_hit_collections = {
            "VXDBarrelHits": None,
            "ITBarrelHits": None,
            "OTBarrelHits": None,
            "VXDEndcapHits": None,
            "ITEndcapHits": None,
            "OTEndcapHits": None,
        }
        self.tracker_hits_relations = {
            "VXDBarrelHits": "VXDBarrelHitsRelations",
            "ITBarrelHits": "ITBarrelHitsRelations",
            "OTBarrelHits": "OTBarrelHitsRelations",
            "VXDEndcapHits": "VXDEndcapHitsRelations",
            "ITEndcapHits": "ITEndcapHitsRelations",
            "OTEndcapHits": "OTEndcapHitsRelations",
        }
        # reco collection -> fired-pixel collection and raw hit->pixel relations
        self.tracker_hit_pixels = {
            "VXDBarrelHits": "VXDBarrelPixels",
            "ITBarrelHits": "ITBarrelPixels",
            "OTBarrelHits": "OTBarrelPixels",
            "VXDEndcapHits": "VXDEndcapPixels",
            "ITEndcapHits": "ITEndcapPixels",
            "OTEndcapHits": "OTEndcapPixels",
        }
        self.tracker_raw_relations = {
            "VXDBarrelHits": "VXDBarrelRawHitRelations",
            "ITBarrelHits": "ITBarrelRawHitRelations",
            "OTBarrelHits": "OTBarrelRawHitRelations",
            "VXDEndcapHits": "VXDEndcapRawHitRelations",
            "ITEndcapHits": "ITEndcapRawHitRelations",
            "OTEndcapHits": "OTEndcapRawHitRelations",
        }
        # reco collection -> name under which its CellIDEncoding is stored in metadata
        self.tracker_hits_encoding = {
            "VXDBarrelHits": "VertexBarrelCollection",
            "ITBarrelHits": "InnerTrackerBarrelCollection",
            "OTBarrelHits": "OuterTrackerBarrelCollection",
            "VXDEndcapHits": "VertexEndcapCollection",
            "ITEndcapHits": "InnerTrackerEndcapCollection",
            "OTEndcapHits": "OuterTrackerEndcapCollection",
        }

        self.cols_hits = ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight", "ch0", "ch1", "value"]
        self.cols_particles = ["particle_id", "vx", "vy", "vz", "px", "py", "pz", "q", "nhits"]
        if self.timing:
            self.cols_hits = ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight", "ch0", "ch1", "value", "t", "tt"]
            self.cols_particles = ["particle_id", "vx", "vy", "vz", "vt", "px", "py", "pz", "q", "nhits"]

        self.cols_hits_csv = ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"]
        self.cols_truth_csv = ["hit_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"]
        self.cols_cells_csv = ["hit_id", "ch0", "ch1", "value"]
        self.cols_particles_csv = ["particle_id", "vx", "vy", "vz", "px", "py", "pz", "q", "nhits"]
        self.cols_detector_csv = ["cell_id", "volume_id", "layer_id", "module_id", "sensor_id"]

        if self.timing:
            self.cols_hits_csv = ["hit_id", "x", "y", "z", "t", "volume_id", "layer_id", "module_id"]
            self.cols_truth_csv = ["hit_id", "particle_id", "tx", "ty", "tz", "tt", "tpx", "tpy", "tpz", "weight"]
            self.cols_particles_csv = ["particle_id", "vx", "vy", "vz", "vt", "px", "py", "pz", "q", "nhits"]

        if self.pixel_cluster_features:
            self.cols_hits += ["npix", "ext_u", "ext_v"]
            self.cols_hits_csv += ["npix", "ext_u", "ext_v"]

        self.surfaces = hit_features.load_surfaces(surfaces) if surfaces else None
        if self.surfaces is not None:
            self.cols_hits_csv += hit_features.feature_names(self.timing, self.pixel_cluster_features)

        self.df_detector = pd.DataFrame(columns=self.cols_detector_csv)

        self.particle_dict_bkg = {
            "particle_id": 0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "px": 0.0,
            "py": 0.0,
            "pz": 0.0,
            "q": 0.0,
            "nhits": 1
        }
        if self.timing:
            self.particle_dict_bkg["vt"] = 0.0

    # ---- EDM4hep helpers (replace LCIO reader/decoder/relation-navigator) ----

    def _make_reader(self):
        return root_io.Reader(self.input_files if len(self.input_files) > 1 else self.input_files[0])

    def _build_decoders(self, reader):
        """Read the CellID encoding strings from the metadata frame and build a
        BitFieldCoder per reco tracker-hit collection."""
        decoders = {}
        meta = reader.get("metadata")[0]
        for col in self.tracker_hit_collections.keys():
            encoding = meta.get_parameter(self.tracker_hits_encoding[col] + "__CellIDEncoding")
            decoders[col] = BitFieldCoder(encoding)
        return decoders

    @staticmethod
    def _relation_map(relation_collection):
        """Build a map reco-hit-index -> first related SimTrackerHit, mirroring
        LCRelationNavigator.getRelatedToObjects(hit)[0]."""
        rel_map = {}
        for link in relation_collection:
            idx = link.getFrom().getObjectID().index
            if idx not in rel_map:  # keep first relation, as in the LCIO version
                rel_map[idx] = link.getTo()
        return rel_map

    @staticmethod
    def _relation_multimap(relation_collection):
        """Build map from source index to all related target objects."""
        rel_map = {}
        for link in relation_collection:
            idx = link.getFrom().getObjectID().index
            rel_map.setdefault(idx, []).append(link.getTo())
        return rel_map

    def process(self, start_event=0, end_event=None):
        n_events = end_event - start_event if end_event is not None else -1

        reader = self._make_reader()
        decoders = self._build_decoders(reader)
        events = reader.get("events")

        for ievt, event in enumerate(events):
            if end_event is not None and ievt >= end_event:
                self.logger.info("Stop processing events at event " + str(ievt))
                break
            if ievt < start_event:
                continue
            if ievt == start_event:
                self.logger.info("Start processing events at event " + str(ievt))

            if (ievt % 100 == 0 and n_events == -1) or (n_events > -1 and n_events < 100) or (n_events > 100 and (ievt - start_event) % (n_events // 10) == 0) or ievt == end_event - 1:
                self.logger.info(f"Processing event {str(ievt)} ({ievt - start_event + 1}/{n_events if n_events > -1 else 'all'})")

            # ---- gather all hits of the event into flat columns (single O(n) pass) ----
            # Instead of growing df_hits/df_particles row by row (O(n^2)), we collect
            # plain lists and build the dataframes once, then assign particle_ids and
            # the particle table with vectorized numpy/pandas operations.
            rec = {k: [] for k in ["x", "y", "z", "volume_id", "layer_id", "module_id", "subdet",
                                   "tx", "ty", "tz", "tpx", "tpy", "tpz",
                                   "ch0", "ch1", "value", "t", "tt"]}
            if self.pixel_cluster_features:
                rec["npix"] = []
                rec["ext_u"] = []
                rec["ext_v"] = []
            avail_l, is_sim_l = [], []
            px_l, py_l, pz_l, vx_l, vy_l, vz_l, q_l, vt_l = [], [], [], [], [], [], [], []

            for col, col_i in zip(self.tracker_hit_collections.keys(), range(len(self.tracker_hit_collections.keys()))):
                hit_collection = event.get(col)
                relation = self._relation_map(event.get(self.tracker_hits_relations[col]))

                pixel_rel = None
                if self.pixel_cluster_features:
                    pixel_rel = self._relation_multimap(event.get(self.tracker_raw_relations[col]))

                self.logger.info(f"Processing hit collection: {col} ({col_i+1}/{len(self.tracker_hit_collections.keys())}), number of hits: {len(hit_collection)}")

                # keep only hits that have a related sim hit, preserving order
                pairs = []
                for hit in hit_collection:
                    simhit = relation.get(hit.getObjectID().index)
                    if simhit is not None:
                        pairs.append((hit, simhit))
                if not pairs:
                    continue

                # vectorized cellID decode for this collection
                cid = np.fromiter((h.getCellID() for h, _ in pairs), dtype=np.int64, count=len(pairs))
                vol = decoders[col].value_array("system", cid)
                lay = decoders[col].value_array("layer", cid)
                mod = decoders[col].value_array("module", cid)
                sen = decoders[col].value_array("sensor", cid)
                rec["subdet"].extend([col[:-len("Hits")]] * len(pairs))

                for k, (hit, simhit) in enumerate(pairs):
                    pos = hit.getPosition()
                    spos = simhit.getPosition()
                    smom = simhit.getMomentum()
                    rec["x"].append(pos[0]); rec["y"].append(pos[1]); rec["z"].append(pos[2])
                    rec["volume_id"].append(int(vol[k])); rec["layer_id"].append(int(lay[k])); rec["module_id"].append(int(mod[k]))
                    rec["tx"].append(spos[0]); rec["ty"].append(spos[1]); rec["tz"].append(spos[2])
                    rec["tpx"].append(smom[0]); rec["tpy"].append(smom[1]); rec["tpz"].append(smom[2])
                    rec["ch0"].append(int(cid[k])); rec["ch1"].append(int(sen[k]))
                    rec["value"].append(hit.getEDep())
                    rec["t"].append(hit.getTime()); rec["tt"].append(simhit.getTime())

                    if self.pixel_cluster_features:
                        pix = pixel_rel.get(hit.getObjectID().index, []) if pixel_rel is not None else []
                        rec["npix"].append(int(len(pix)))
                        if pix:
                            u = [p.getPosition()[0] for p in pix]
                            v = [p.getPosition()[1] for p in pix]
                            rec["ext_u"].append(int(round(max(u) - min(u))) + 1)
                            rec["ext_v"].append(int(round(max(v) - min(v))) + 1)
                        else:
                            rec["ext_u"].append(0)
                            rec["ext_v"].append(0)

                    mc_particle = simhit.getParticle()
                    if mc_particle.isAvailable():
                        mom = mc_particle.getMomentum(); ver = mc_particle.getVertex()
                        avail_l.append(True)
                        is_sim_l.append(bool(mc_particle.isCreatedInSimulation()))
                        px_l.append(mom[0]); py_l.append(mom[1]); pz_l.append(mom[2])
                        vx_l.append(ver[0]); vy_l.append(ver[1]); vz_l.append(ver[2])
                        q_l.append(mc_particle.getCharge()); vt_l.append(mc_particle.getTime())
                    else:  # mc_particle is null (= background / noise hit)
                        avail_l.append(False); is_sim_l.append(False)
                        px_l.append(np.nan); py_l.append(np.nan); pz_l.append(np.nan)
                        vx_l.append(np.nan); vy_l.append(np.nan); vz_l.append(np.nan)
                        q_l.append(np.nan); vt_l.append(np.nan)

            n = len(rec["x"])
            avail = np.array(avail_l, dtype=bool)
            is_sim = np.array(is_sim_l, dtype=bool)  # already False where not available
            px = np.array(px_l); py = np.array(py_l); pz = np.array(pz_l)
            vx = np.array(vx_l); vy = np.array(vy_l); vz = np.array(vz_l)
            q = np.array(q_l); vt = np.array(vt_l)

            # ---- assign particle_id per hit (vectorized) ----
            # background / noise -> 0; sim particles mapped to bkg -> 0; otherwise a
            # running id by first appearance of the (px, py, pz) momentum triple.
            map_to_bkg = is_sim & self.sim_particles_to_bkg
            real = avail & ~map_to_bkg
            particle_id_arr = np.zeros(n, dtype=np.int64)
            real_idx = np.flatnonzero(real)
            if real_idx.size:
                keys = list(zip(px[real_idx].tolist(), py[real_idx].tolist(), pz[real_idx].tolist()))
                codes = pd.factorize(pd.Series(keys), sort=False)[0]  # 0,1,2,... in order of appearance
                particle_id_arr[real_idx] = codes + 1

            weight = np.where(is_sim, 1.0, 0.0)

            # ---- build df_hits in one shot (float dtype, matching the original output) ----
            df_hits = pd.DataFrame({
                "hit_id": np.arange(n, dtype=np.float64),
                "x": rec["x"], "y": rec["y"], "z": rec["z"],
                "volume_id": rec["volume_id"], "layer_id": rec["layer_id"], "module_id": rec["module_id"],
                "particle_id": particle_id_arr,
                "tx": rec["tx"], "ty": rec["ty"], "tz": rec["tz"],
                "tpx": rec["tpx"], "tpy": rec["tpy"], "tpz": rec["tpz"],
                "weight": weight,
                "ch0": rec["ch0"], "ch1": rec["ch1"], "value": rec["value"],
                "t": rec["t"], "tt": rec["tt"],
            }).astype(np.float64)

            if self.pixel_cluster_features:
                df_hits["npix"] = np.asarray(rec["npix"], dtype=np.float64)
                df_hits["ext_u"] = np.asarray(rec["ext_u"], dtype=np.float64)
                df_hits["ext_v"] = np.asarray(rec["ext_v"], dtype=np.float64)

            if self.surfaces is not None:
                geo_feats = hit_features.compute_hit_features(
                    self.surfaces, rec["subdet"], df_hits["ch0"].to_numpy(), df_hits[["x", "y", "z"]].to_numpy(),
                    t=df_hits["t"].to_numpy() if self.timing else None,
                    ext_u=df_hits["ext_u"].to_numpy() if self.pixel_cluster_features else None,
                    ext_v=df_hits["ext_v"].to_numpy() if self.pixel_cluster_features else None,
                )
                for name, values in geo_feats.items():
                    df_hits[name] = values
                n_unmatched = int(np.isnan(geo_feats["inc_cos"]).sum())
                if n_unmatched:
                    self.logger.warning(f"{n_unmatched} hits have no sensor in the surfaces table; their geometry features are NaN")

            # ---- build df_particles (vectorized) ----
            idx = np.arange(n)
            frames = []
            if real_idx.size:
                real_tbl = pd.DataFrame({
                    "particle_id": particle_id_arr[real_idx],
                    "vx": vx[real_idx], "vy": vy[real_idx], "vz": vz[real_idx], "vt": vt[real_idx],
                    "px": px[real_idx], "py": py[real_idx], "pz": pz[real_idx], "q": q[real_idx],
                    "_idx": idx[real_idx],
                })
                g = real_tbl.groupby("particle_id", sort=False)
                parts = g.agg(
                    vx=("vx", "first"), vy=("vy", "first"), vz=("vz", "first"), vt=("vt", "first"),
                    px=("px", "first"), py=("py", "first"), pz=("pz", "first"), q=("q", "first"),
                    nhits=("particle_id", "size"), _idx=("_idx", "first"),
                ).reset_index()
                frames.append(parts)

            # background particle (id 0): sim-mapped-to-bkg hits always contribute, noise
            # hits only if noise_to_bkg; kinematics come from the first contributing hit.
            id0_contrib = map_to_bkg | ((~avail) & self.noise_to_bkg)
            if id0_contrib.any():
                first0 = int(np.flatnonzero(id0_contrib)[0])
                nh0 = int(id0_contrib.sum())
                if avail[first0]:  # first contributor is a sim particle mapped to bkg
                    row0 = {"particle_id": 0, "vx": vx[first0], "vy": vy[first0], "vz": vz[first0],
                            "vt": vt[first0], "px": px[first0], "py": py[first0], "pz": pz[first0],
                            "q": q[first0], "nhits": nh0, "_idx": first0}
                else:  # first contributor is a noise hit -> generic background kinematics
                    row0 = dict(self.particle_dict_bkg)
                    row0.setdefault("vt", 0.0)
                    row0["nhits"] = nh0
                    row0["_idx"] = first0
                frames.append(pd.DataFrame([row0]))

            cols_particles_full = ["particle_id", "vx", "vy", "vz", "vt", "px", "py", "pz", "q", "nhits"]
            if frames:
                df_particles = pd.concat(frames, ignore_index=True)
                df_particles = df_particles.sort_values("_idx").drop(columns="_idx").reset_index(drop=True)
                df_particles = df_particles[cols_particles_full].astype(np.float64)
            else:
                df_particles = pd.DataFrame(columns=cols_particles_full)

            # remove duplicates in df_particles
            df_particles = df_particles.drop_duplicates(subset=["particle_id", "vx", "vy", "vz", "vt", "px", "py", "pz", "q"], keep="first").reset_index(drop=True)

            # add to detector dataframe with ch1 as cell_id
            df_det = df_hits[["ch1", "volume_id", "layer_id", "module_id", "ch0"]].drop_duplicates().reset_index(drop=True).rename(columns={"ch1": "sensor_id", "ch0": "cell_id"})
            self.df_detector = pd.concat([self.df_detector, df_det], ignore_index=True).drop_duplicates().reset_index(drop=True)

            # save to csv
            df_hits.to_csv(os.path.join(self.output_dir, f"event{ievt:09d}-hits.csv"), index=False, header=True, columns=self.cols_hits_csv)
            df_hits.to_csv(os.path.join(self.output_dir, f"event{ievt:09d}-truth.csv"), index=False, header=True, columns=self.cols_truth_csv)
            df_hits.to_csv(os.path.join(self.output_dir, f"event{ievt:09d}-cells.csv"), index=False, header=True, columns=self.cols_cells_csv)

            df_particles.to_csv(os.path.join(self.output_dir, f"event{ievt:09d}-particles.csv"), index=False, header=True, columns=self.cols_particles_csv)

            if self.seeds:
                self._write_seeds(df_hits, ievt)

    def _write_seeds(self, df_hits, ievt):
        # Truth-tracklet seeds with fakes and duplicates; hit_id_* refer to the hit_id column of the hits csv.
        seeds = seed_utils.generate_seeds(
            df_hits[["x", "y", "z"]].to_numpy(),
            df_hits["particle_id"].to_numpy().astype(np.int64),
            t=df_hits["t"].to_numpy() if self.timing else None,
            rng=np.random.default_rng([self.seed_random_seed, ievt]),
            **self.seed_options,
        )
        hit_id = df_hits["hit_id"].to_numpy().astype(np.int64)
        df_seeds = pd.DataFrame({
            "seed_id": np.arange(len(seeds["hits"])),
            **{f"hit_id_{i}": hit_id[seeds["hits"][:, i]] for i in range(3)},
            **{name: seeds["params"][:, i] for i, name in enumerate(seed_utils.PARAM_NAMES)},
            "particle_id": seeds["particle_id"],
            "purity": seeds["purity"],
            "is_true": seeds["is_true"].astype(np.int64),
            "origin": seeds["origin"],
        })
        df_seeds.to_csv(os.path.join(self.output_dir, f"event{ievt:09d}-seeds.csv"), index=False)
    def collect_detector_info(self):
        # separate function to only collect detector information for detector csv.
        # Detector geometry depends only on the reco hit cellID (which encodes
        # system/side/layer/module/sensor), so we gather the unique cellIDs per
        # collection and decode them vectorized at the end instead of growing a
        # DataFrame row by row (which is O(n^2)).
        reader = self._make_reader()
        decoders = self._build_decoders(reader)
        events = reader.get("events")

        # one set of unique cellIDs per collection (uniques ~ number of sensors)
        cell_ids = {col: set() for col in self.tracker_hit_collections.keys()}

        for ievt, event in enumerate(events):
            # if ievt % 100 == 0:
            self.logger.info("Event " + str(ievt))

            for col in self.tracker_hit_collections.keys():
                hit_collection = event.get(col)
                cell_ids[col].update(hit.getCellID() for hit in hit_collection)

        frames = []
        for col, ids in cell_ids.items():
            if not ids:
                continue
            cid = np.fromiter(ids, dtype=np.int64, count=len(ids))
            frames.append(pd.DataFrame({
                "cell_id": cid,
                "volume_id": decoders[col].value_array("system", cid),
                "layer_id": decoders[col].value_array("layer", cid),
                "module_id": decoders[col].value_array("module", cid),
                "sensor_id": decoders[col].value_array("sensor", cid),
            }))

        self.df_detector = (
            pd.concat([self.df_detector] + frames, ignore_index=True)
              .drop_duplicates()
              .reset_index(drop=True)
        )

    def export_detector_csv(self):
        self.df_detector = self.df_detector.drop_duplicates(subset=["volume_id", "layer_id", "module_id", "sensor_id"], keep="first").reset_index(drop=True)
        self.df_detector.sort_values(by=["volume_id", "layer_id", "module_id", "sensor_id"], inplace=True)
        self.df_detector.to_csv(os.path.join(os.path.dirname(self.output_dir), "detectors.csv"), index=False, header=True, columns=self.cols_detector_csv)

        print("Volume IDs:", self.df_detector["volume_id"].unique())
        print("Layer IDs:", self.df_detector["layer_id"].unique())
        print("Module IDs:", self.df_detector["module_id"].unique())
        print("Sensor IDs:", self.df_detector["sensor_id"].unique())

    def update_detector_csv(self):
        if not os.path.exists(os.path.join(os.path.dirname(self.output_dir), "detectors.csv")):
            return

        existing_df = pd.read_csv(os.path.join(os.path.dirname(self.output_dir), "detectors.csv"))
        self.df_detector = pd.concat([existing_df, self.df_detector]).drop_duplicates().reset_index(drop=True)
        self.df_detector.sort_values(by=["volume_id", "layer_id", "module_id", "sensor_id"], inplace=True)
        self.df_detector.to_csv(os.path.join(os.path.dirname(self.output_dir), "detectors.csv"), index=False, header=True, columns=self.cols_detector_csv)

    def plot_hits_per_event(self, by_pt=True, by_subdetector=True, by_barrel_endcap=True, n_events=1000, n_bins=20):
        muon_pts = []
        n_hits_per_subdetector_list = []

        reader = self._make_reader()
        events = reader.get("events")

        for ievt, event in enumerate(events):
            if ievt >= n_events:
                break
            if ievt % 100 == 0:
                self.logger.info("Event " + str(ievt))

            n_hits_per_subdetector = {col: 0 for col in self.tracker_hit_collections.keys()}
            muon_pt = 0.0

            for col, col_i in zip(self.tracker_hit_collections.keys(), range(len(self.tracker_hit_collections.keys()))):
                hit_collection = event.get(col)
                relation_collection = event.get(self.tracker_hits_relations[col])
                relation = self._relation_map(relation_collection)

                n_hits_per_subdetector[col] = len(hit_collection)

                for i in range(len(hit_collection)):
                    hit = hit_collection[i]
                    simhit = relation.get(hit.getObjectID().index)
                    if simhit is None:
                        continue

                    mc_particle = simhit.getParticle()
                    if mc_particle.isAvailable():
                        if abs(mc_particle.getPDG()) == 13:  # muon
                            p = mc_particle.getMomentum()
                            muon_pt = np.sqrt(p[0]**2 + p[1]**2)

            muon_pts.append(muon_pt)
            n_hits_per_subdetector_list.append(n_hits_per_subdetector)

        # plotting
        n_hits_per_subdetector = np.array([list(d.values()) for d in n_hits_per_subdetector_list])
        n_hits_total = n_hits_per_subdetector.sum(axis=1)
        muon_pts = np.array(muon_pts)

        fig, axs = plt.subplots(2 if by_barrel_endcap else 1, 3 if by_subdetector else 1, figsize=(5 * (3 if by_subdetector else 1), 5 * (2 if by_barrel_endcap else 1)))

        if not (by_barrel_endcap and by_subdetector):
            axs = axs[:, np.newaxis] if by_subdetector else axs[np.newaxis, :]

        n_hits_barrel = n_hits_per_subdetector[:, [0, 2, 4]].sum(axis=1)
        n_hits_endcap = n_hits_per_subdetector[:, [1, 3, 5]].sum(axis=1)

        n_hits_vertex = n_hits_per_subdetector[:, [0, 1]].sum(axis=1)
        n_hits_inner = n_hits_per_subdetector[:, [2, 3]].sum(axis=1)
        n_hits_outer = n_hits_per_subdetector[:, [4, 5]].sum(axis=1)

        n_hits_total = n_hits_per_subdetector.sum(axis=1)

        for i in range(2 if by_barrel_endcap else 1):  # by_barrel_endcap
            for j in range(3 if by_subdetector else 1):  # by_subdetector
                ax = axs[i, j]

                if by_barrel_endcap and by_subdetector:
                    n_hits = n_hits_per_subdetector[:, i*3 + j]
                elif by_barrel_endcap and not by_subdetector:
                    n_hits = n_hits_barrel if i == 0 else n_hits_endcap
                elif not by_barrel_endcap and by_subdetector:
                    n_hits = n_hits_vertex if j == 0 else (n_hits_inner if j == 1 else n_hits_outer)
                else:
                    n_hits = n_hits_total

                if by_pt:
                    bins = np.linspace(np.min(muon_pts), np.max(muon_pts), n_bins)
                    n_hits_binned = np.zeros(n_bins - 1)
                    for k in range(n_bins - 1):
                        in_bin = (muon_pts >= bins[k]) & (muon_pts < bins[k+1])
                        n_hits_binned[k] = np.mean(n_hits[in_bin]) if np.sum(in_bin) > 0 else 0
                    ax.errorbar(0.5 * (bins[:-1] + bins[1:]), n_hits_binned, yerr=np.sqrt(n_hits_binned), fmt='x', capsize=5)
                else:
                    bins = np.linspace(np.min(n_hits), np.max(n_hits), n_bins)
                    n_hits_binned, _ = np.histogram(n_hits, bins=bins)
                    ax.bar(0.5 * (bins[:-1] + bins[1:]), n_hits_binned, width=(bins[1]-bins[0]))

                title = ""
                if by_barrel_endcap:
                    title += "Barrel " if i == 0 else "Endcap "
                if by_subdetector:
                    title += "Vertex " if j == 0 else ("Inner " if j == 1 else "Outer ")
                ax.set_title(title + "Hits" if title != "" else "Hits")
                ax.set_xlabel("Muon pT [GeV]" if by_pt else "Number of hits")
                ax.set_ylabel("Average number of hits per event" if by_pt else "Number of events")
                ax.grid()

        fig.suptitle("Number of hits per event" + (f" vs muon pT ({n_events} events)" if by_pt else f" distribution ({n_events} events)"), fontsize=16)

        plt.tight_layout()
        plt.savefig(f"{os.path.dirname(self.output_dir)}/nhits{'_bypt' if by_pt else ''}{'_bysubdet' if by_subdetector else ''}{'_bybarrelendcap' if by_barrel_endcap else ''}.png")
        print(f"saved as {os.path.dirname(self.output_dir)}/nhits{'_bypt' if by_pt else ''}{'_bysubdet' if by_subdetector else ''}{'_bybarrelendcap' if by_barrel_endcap else ''}.png")


if __name__ == "__main__":
    input_dir = "/data_in"
    output_dir = "/data_/acorn/muonGun_pT_0_50/extractor"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    extractor = edm4hep_extractor(input_dir, output_dir, timing=True, coned_hits=True, logger=logger)
    # extractor.process()
    # extractor.export_detector_csv()
    extractor.plot_hits_per_event(n_events=1000, n_bins=20)
