import os
from typing import Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy
import pandas
from sklearn.metrics import r2_score, roc_auc_score
import xgboost

from track import Run
from meta_boosting import train_booster_on_array, load_selector, get_model_path
from meta_boosting.model_utils import Data, XGBoostModel, LightGBMModel
from utils.phenotype import load_phenotype_for_xgboost
from meta_boosting.xgb_utils import get_arrays_new


def score(y_true: numpy.ndarray, y_pred: numpy.ndarray, fc: int, prefix: str, is_clf: bool):
    if not is_clf:
        m = r2_score(y_true, y_pred)
        print(f'{prefix} r2: {m:.4f} for {fc} total features')
    else:
        m = roc_auc_score(y_true, y_pred)
        print(f'mean of y_pred on {prefix} dataset is {numpy.mean(y_pred):.5f}')
        print(f'{prefix} auc: {m:.4f} for {fc} total features')


def load_gwas_data() -> Tuple[Data, DictConfig]:
    pass


def load_xgbsel_data(run: Run, cfg: DictConfig) -> Tuple[Data, DictConfig]:
    fc = cfg.experiment.meta.pred_feature_count
    pheno_data = load_phenotype_for_xgboost(cfg)

    X_train, X_val, X_test, y_train, y_val = get_arrays_new(
            pheno_data.train,
            pheno_data.test,
            run.results_path('selection_temp.npz'),
            cfg.dataset.genotype.train,
            cfg.dataset.genotype.val,
            cfg.dataset.genotype.test,
            run.chromosomes,
            is_clf=False,
            snp_count=fc,
            feature_data=pheno_data.features,
            prefix='les'
    )
    
    cov_count = X_train.shape[1] - fc
    print(f'we have cov_count: {cov_count}')

    model_path = run.model_path('main_10k_md2_alpha20.0.model')
    model_cfg = OmegaConf.create({'model_path': model_path, 'is_clf': False})
    model_cfg = OmegaConf.merge(model_cfg, cfg.experiment.pred_model)
    
    data = Data(X_train, X_val, X_test, y_train, y_val, None, cov_count)
    # data = data.standardize()
    return data, model_cfg


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
    pheno_data = load_phenotype_for_xgboost(cfg)
    if cfg.experiment.xgbsel:
        data, model_cfg = load_xgbsel_data(run, cfg)
    else:
        data, model_cfg = load_gwas_data()

    print(numpy.unique(data.X_test.values, return_counts=True))

    if cfg.experiment.pred_model.name == 'xgboost':
        model = XGBoostModel(model_cfg)
    else:
        raise ValueError('model name should be xgboost')
    
    model.load_model_from_path()
    y_val_pred, y_test_pred = model.predict(data)

    score(data.y_val, y_val_pred, data.X_train.shape[1], 'val', model_cfg.is_clf)

    y_test_pred = pandas.DataFrame(data=y_test_pred, columns=['y_test_pred'], index=data.X_test.index)
    print(f'We predicted {y_test_pred.shape[0]} rows from {data.X_test.shape} rows and features')
    y = pheno_data.test.merge(y_test_pred, how='inner', left_index=True, right_index=True)
    print(f'After merge with loaded test data with shape {pheno_data.test.shape} we have {y.shape[0]} rows to eval')
    
    y.loc[:, 'trait'] = data.scaler.transform(y.trait.values.reshape(-1, 1))
    score(y.trait.values, y.y_test_pred.values, data.X_train.shape[1], 'test', model_cfg.is_clf)


if __name__ == '__main__':
    run()