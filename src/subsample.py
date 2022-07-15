import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy

from track import Run
from meta_boosting import train_booster_on_array, load_selector, get_model_path


@hydra.main(config_path="configs", config_name="default")
def run(cfg: DictConfig):
    snapshot = cfg.experiment.snapshot
    print(f'cfg: {cfg}')
    print(f'loading run with snapshot {snapshot}')
    run = Run(cfg.dataset.run_dir, cfg.experiment.phenotype.name, snapshot=snapshot)
    run.chromosomes = ['MAIN']
    description = {
        'phenotype': OmegaConf.to_container(cfg.experiment.phenotype),
        'output_dir': os.getcwd()
    }
    run.save_params(description, OmegaConf.to_container(cfg.experiment))

    selector = load_selector(run)
    fc = cfg.experiment.meta.pred_feature_count
    X_train, X_val, X_test, y_train, y_val = selector.transform(fc, add_features=True)
    cov_count = X_train.shape[1] - fc
    print(f'we have cov_count: {cov_count}')

    numpy.savez('/beegfs/home/a.medvedev/xgboost_selection/outputs/to_save.npz', X_train=X_train[:100000], y_train=y_train[:100000])
    print(f'saved x-train and y-train')

if __name__ == '__main__':
    run()