import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy
import random
print(os.environ['PYTHONPATH'])
print(os.getcwd())

from track import Run
from meta_boosting import XGBSelector
from utils.phenotype import load_phenotype_for_xgboost


@hydra.main(config_path="configs", config_name="default")
def run_selection(cfg: DictConfig):
    print('running with cfg:')
    print(cfg)

    run = Run(cfg.dataset.run_dir, cfg.experiment.phenotype.name)
    run.chromosomes = ['MAIN']
    description = {
        'phenotype': OmegaConf.to_container(cfg.experiment.phenotype),
        'output_dir': os.getcwd()
    }
    run.save_params(description, OmegaConf.to_container(cfg.experiment.params))
    print(f'created run {run}')
    random.seed(cfg.experiment.random_state)
    numpy.random.seed(cfg.experiment.random_state)

    pheno_data = load_phenotype_for_xgboost(cfg)
    selector = XGBSelector(
        cfg.dataset.genotype.train,
        cfg.dataset.genotype.val,
        cfg.dataset.genotype.test,
        pheno_data.train,
        pheno_data.test,
        run,
        feature_data=pheno_data.features,
        snp_window_len=cfg.experiment.params.meta.snp_window_len,
        window_trees=cfg.experiment.params.meta.window_trees, 
        verbose=True,
        m_eta=1.0,
        deterministic_histogram=False,
        reverse_direction=(cfg.experiment.params.meta.direction != 'forward'),
        is_clf=(cfg.experiment.phenotype.type != 'real'),
        tree_method='gpu_hist',
        early_stopping_rounds=20,
        **cfg.experiment.params.xgbsel
    )
    
    selector.fit()


if __name__ == '__main__':
    run_selection()

