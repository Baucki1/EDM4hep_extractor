import numpy as np
import pandas as pd

import hit_features as hf

SURFACES = pd.DataFrame({
    "subdet": ["ITBarrel", "ITBarrel"],
    "cellID": [7, 8],
    "ux": [0.0, 0.0], "uy": [1.0, 1.0], "uz": [0.0, 0.0],
    "vx": [0.0, 0.0], "vy": [0.0, 0.0], "vz": [1.0, 1.0],
    "nx": [1.0, 1.0], "ny": [0.0, 0.0], "nz": [0.0, 0.0],
})


def test_radial_hit_is_perpendicular_and_matches_cluster_model():
    f = hf.compute_hit_features(
        SURFACES, ["ITBarrel"], [7], np.array([[30.0, 0.0, 0.0]]),
        t=np.array([30.0 / hf.C_MM_PER_NS]), ext_u=np.array([5.0]), ext_v=np.array([1.0]),
    )
    assert np.isclose(f["inc_cos"][0], 1.0)
    assert np.isclose(f["tan_u_ip"][0], 0.0) and np.isclose(f["tan_v_ip"][0], 0.0)
    assert np.isclose(f["t_corr"][0], 0.0)
    # 0.1 mm * |0 - 0.8| / 0.025 mm + 1
    assert np.isclose(f["ext_u_exp"][0], 4.2) and np.isclose(f["ext_v_exp"][0], 1.0)
    assert np.isclose(f["ext_u_res"][0], 0.8) and np.isclose(f["ext_v_res"][0], 0.0)


def test_tilted_hit_and_long_pixel_v_pitch():
    f = hf.compute_hit_features(
        SURFACES, ["ITBarrel"], [8], np.array([[30.0, 30.0, 30.0]]),
        ext_u=np.array([1.0]), ext_v=np.array([1.0]),
    )
    assert np.isclose(f["tan_u_ip"][0], 1.0) and np.isclose(f["tan_v_ip"][0], 1.0)
    assert np.isclose(f["ext_u_exp"][0], 0.1 * 0.2 / 0.025 + 1)
    assert np.isclose(f["ext_v_exp"][0], 0.1 * 1.0 / 0.312 + 1)


def test_unknown_sensor_gives_nan_and_keeps_order():
    pos = np.array([[30.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    f = hf.compute_hit_features(SURFACES, ["ITBarrel", "ITBarrel"], [999, 7], pos)
    assert np.isnan(f["inc_cos"][0]) and np.isclose(f["inc_cos"][1], 1.0)
    assert "t_corr" not in f and "ext_u_res" not in f


if __name__ == "__main__":
    test_radial_hit_is_perpendicular_and_matches_cluster_model()
    test_tilted_hit_and_long_pixel_v_pitch()
    test_unknown_sensor_gives_nan_and_keeps_order()
    print("hit_features tests passed")
