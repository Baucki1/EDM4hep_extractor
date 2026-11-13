"""Per-hit features from the sensor geometry: IP-pointing incidence angles and corrected hit time."""
import numpy as np
import pandas as pd

C_MM_PER_NS = 299.792458
PITCH_U_MM = 0.025
TAN_LORENTZ = 0.8
MAX_ABS_TAN = 20.0
# Must match the MuonCVXDDigitiser configuration that produced the fired pixels (see surfaces/clusters.py).
THICKNESS_MM = {"VXDBarrel": 0.05, "VXDEndcap": 0.05, "ITBarrel": 0.1, "ITEndcap": 0.1, "OTBarrel": 0.1, "OTEndcap": 0.1}
PITCH_V_MM = {"VXDBarrel": 0.025, "VXDEndcap": 0.025, "ITBarrel": 0.312, "ITEndcap": 0.312, "OTBarrel": 0.312, "OTEndcap": 0.312}
SURFACE_COLUMNS = ["subdet", "cellID", "ux", "uy", "uz", "vx", "vy", "vz", "nx", "ny", "nz"]


def load_surfaces(path):
    return pd.read_csv(path, usecols=SURFACE_COLUMNS).drop_duplicates(subset=["subdet", "cellID"])


def feature_names(timing, pixel_cluster_features):
    names = ["inc_cos", "tan_u_ip", "tan_v_ip"]
    if timing:
        names.append("t_corr")
    if pixel_cluster_features:
        names += ["ext_u_exp", "ext_v_exp", "ext_u_res", "ext_v_res"]
    return names


def compute_hit_features(surfaces, subdet, cell_id, pos, t=None, ext_u=None, ext_v=None):
    """Features of a straight track from the origin through each hit; NaN where the sensor is not in `surfaces`.

    The expected cluster extents use the same geometric model as surfaces/clusters.py, so the residuals
    ext_*_res measure how far a cluster is from what an IP-pointing track would leave.
    """
    keys = pd.DataFrame({"subdet": np.asarray(subdet), "cellID": np.asarray(cell_id).astype(np.int64)})
    geo = keys.merge(surfaces, on=["subdet", "cellID"], how="left")
    u = geo[["ux", "uy", "uz"]].to_numpy(dtype=float)
    v = geo[["vx", "vy", "vz"]].to_numpy(dtype=float)
    n = geo[["nx", "ny", "nz"]].to_numpy(dtype=float)

    pos = np.asarray(pos, dtype=float)
    dist = np.linalg.norm(pos, axis=1)
    r_hat = pos / np.maximum(dist, 1e-9)[:, None]
    r_n = np.einsum("ij,ij->i", r_hat, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_u = np.clip(np.einsum("ij,ij->i", r_hat, u) / r_n, -MAX_ABS_TAN, MAX_ABS_TAN)
        tan_v = np.clip(np.einsum("ij,ij->i", r_hat, v) / r_n, -MAX_ABS_TAN, MAX_ABS_TAN)

    feats = {"inc_cos": np.abs(r_n), "tan_u_ip": tan_u, "tan_v_ip": tan_v}
    if t is not None:
        feats["t_corr"] = np.asarray(t, dtype=float) - dist / C_MM_PER_NS
    if ext_u is not None and ext_v is not None:
        thickness = pd.Series(np.asarray(subdet)).map(THICKNESS_MM).to_numpy(dtype=float)
        pitch_v = pd.Series(np.asarray(subdet)).map(PITCH_V_MM).to_numpy(dtype=float)
        feats["ext_u_exp"] = thickness * np.abs(tan_u - TAN_LORENTZ) / PITCH_U_MM + 1
        feats["ext_v_exp"] = thickness * np.abs(tan_v) / pitch_v + 1
        feats["ext_u_res"] = np.asarray(ext_u, dtype=float) - feats["ext_u_exp"]
        feats["ext_v_res"] = np.asarray(ext_v, dtype=float) - feats["ext_v_exp"]
    return feats
