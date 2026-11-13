import sys, os
import argparse
import logging

from extractor_edm4hep import edm4hep_extractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hit information for training.")
    parser.add_argument("--input", type=str, help="Input directory containing EDM4hep (.edm4hep.root) files.")
    parser.add_argument("--output", type=str, help="Output directory for extracted data.")
    parser.add_argument("--n_events", type=int, default=1000, help="Number of events to process.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--i_worker", type=int, default=0, help="Worker index for parallel processing.")
    parser.add_argument("--timing", type=bool, default=True, help="Extract hit time information for 4D tracking.")
    parser.add_argument("--sim_particles_to_bkg", type=bool, default=False, help="Map sim-created particles to background particles.")
    parser.add_argument("--noise_to_bkg", type=bool, default=True, help="Map noise hits to background particle (id 0).")
    parser.add_argument("--coned_hits", type=bool, default=False, help="Use coned hits collections.")
    parser.add_argument("--pixel_cluster_features", type=bool, default=True, help="Extract pixel cluster features.")
    parser.add_argument("--surfaces", type=str, default=None, help="surfaces.csv from surfaces/export_surfaces.py; enables IP-pointing and time-of-flight hit features.")
    parser.add_argument("--seeds", action="store_true", help="Also write eventN-seeds.csv with truth-tracklet helix seeds (true, duplicate and fake).")
    parser.add_argument("--seed_min_purity", type=float, default=1.0, help="Fraction of a seed's three hits that must share a particle for the seed to count as true.")
    parser.add_argument("--seed_duplicates", type=int, default=1, help="Extra tracklets per particle, labelled as duplicates.")
    parser.add_argument("--seed_fake_ratio", type=float, default=5.0, help="Fake seeds per primary seed.")
    parser.add_argument("--seed_max_start", type=int, default=3, help="Tracklets start within the first N hit triplets of a particle.")
    parser.add_argument("--seed_random_seed", type=int, default=0, help="Base seed; each event derives its own generator.")
    #parser.add_argument("--fast_mode", type=bool, default=False, help="Fast modes.")

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    n_events = args.n_events
    n_workers = args.n_workers
    i_worker = args.i_worker
    timing = args.timing
    sim_particles_to_bkg = args.sim_particles_to_bkg
    noise_to_bkg = args.noise_to_bkg
    coned_hits = args.coned_hits
    pixel_cluster_features = args.pixel_cluster_features
    #fast_mode = args.fast_mode

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    start_event, end_event = i_worker * n_events // n_workers, (i_worker + 1) * n_events // n_workers

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Starting EDM4hep extractor...")
    extractor = edm4hep_extractor(input_dir,
                                  output_dir,
                                  timing=timing,
                                  sim_particles_to_bkg=sim_particles_to_bkg,
                                  noise_to_bkg=noise_to_bkg,
                                  coned_hits=coned_hits,
                                  pixel_cluster_features=pixel_cluster_features,
                                  surfaces=args.surfaces,
                                  seeds=args.seeds,
                                  seed_min_purity=args.seed_min_purity,
                                  seed_duplicates=args.seed_duplicates,
                                  seed_fake_ratio=args.seed_fake_ratio,
                                  seed_max_start=args.seed_max_start,
                                  seed_random_seed=args.seed_random_seed,
                                  logger=logger)
    extractor.process(start_event, end_event)

    #if i_worker == 1:
    #    try:
    #        extractor.collect_detector_info()
    #        extractor.export_detector_csv()
    #    except KeyboardInterrupt:
    #        extractor.export_detector_csv()

    sys.exit(0)
