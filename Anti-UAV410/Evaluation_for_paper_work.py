# Evaluation_for_ALL_trained.py
from experiments.anti_uav import ExperimentAntiUAV410
from utils.trackers import Trained_Trackers as Trackers

if __name__ == '__main__':
# Configure dataset root and subset
    dataset_path = '/path/to/anti_uav410/'
    subset = 'test'   # 'val' is also possible

    # Configure experiment pipeline
    # result_dir is used for saving times files etc., actual bbox loading uses tracker['path'] as is
    result_dir = "/path/to/Anti-UAV410/Tracking_results"

    experiment = ExperimentAntiUAV410(root_dir=dataset_path,
                                      subset=subset,
                                      result_dir=result_dir)

    # Generate report: save performance.json / state_accuracy_scores.txt / OPE curves (pdf), etc.
    experiment.report(Trackers)
    print("==> DONE: Trained trackers OPE curves & metrics saved.")
