import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy
import pandas
from sklearn.metrics import r2_score, roc_auc_score
import xgboost

from track import Run
from meta_boosting import train_booster_on_array, load_selector, get_model_path
from meta_boosting.model_utils import Data, XGBoostModel, LightGBMModel, CatBoostModel
from utils.phenotype import load_phenotype_for_xgboost


def score(y_true: numpy.ndarray, y_pred: numpy.ndarray, fc: int, prefix: str, is_clf: bool):
    if not is_clf:
        m = r2_score(y_true, y_pred)
        print(f'{prefix} r2: {m:.4f} for {fc} total features')
    else:
        m = roc_auc_score(y_true, y_pred)
        print(f'mean of y_pred on {prefix} dataset is {numpy.mean(y_pred):.5f}')
        print(f'{prefix} auc: {m:.4f} for {fc} total features')


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

    selector = load_selector(run)
    fc = cfg.experiment.meta.pred_feature_count
    X_train, X_val, X_test, y_train, y_val = selector.transform(fc, add_features=True)
    cov_count = X_train.shape[1] - fc
    print(f'we have cov_count: {cov_count}')
    model_path = get_model_path('main', run, fc, cfg.experiment.pred_model, cov_count)
    model_cfg = OmegaConf.create({'model_path': model_path, 'is_clf': selector.is_clf})
    model_cfg = OmegaConf.merge(model_cfg, cfg.experiment.pred_model)

    data = Data(X_train, X_val, X_test, y_train, y_val, None, cov_count)
    data = data.standardize()
    
    if cfg.experiment.pred_model.name == 'xgboost':
        model = XGBoostModel(model_cfg)
    elif cfg.experiment.pred_model.name == 'lightgbm':
        model = LightGBMModel(model_cfg)
    elif cfg.experiment.pred_model.name == 'catboost':
        model = CatBoostModel(model_cfg)
    else:
        raise ValueError('model name should be of the xgboost, lightgbm')
    
    y_val_pred = model.fit(data)

    score(data.y_val, y_val_pred, X_train.shape[1], 'val', selector.is_clf)
    y_pred = model.predict(data)

    y_pred = pandas.DataFrame(data=y_pred, columns=['y_pred'], index=X_test.index)
    print(f'We predicted {y_pred.shape[0]} rows from {X_test.shape} rows and features')
    y = pheno_data.test.merge(y_pred, how='inner', left_index=True, right_index=True)
    print(f'After merge with loaded test data with shape {pheno_data.test.shape} we have {y.shape[0]} rows to eval')
    
    y.loc[:, 'trait'] = data.scaler.transform(y.trait.values.reshape(-1, 1))
    score(y.trait.values, y.y_pred.values, X_train.shape[1], 'test', selector.is_clf)


@hydra.main(config_path="configs", config_name="default")
def run_training(cfg: DictConfig):
    snapshot = cfg.experiment.snapshot
    print(f'loading run with snapshot {snapshot}')
    run = Run(cfg.dataset.run_dir, cfg.experiment.phenotype.name, snapshot=snapshot)
    run.chromosomes = ['MAIN']
    description = {
        'phenotype': OmegaConf.to_container(cfg.experiment.phenotype),
        'output_dir': os.getcwd()
    }
    run.save_params(description, OmegaConf.to_container(cfg.experiment.params))
    pheno_data = load_phenotype_for_xgboost(cfg)

    selector = load_selector(run)
    fc = cfg.experiment.params.meta.pred_feature_count
    X_train, X_val, X_test, y_train, y_val = selector.transform(fc, add_features=True)
    print(f'X shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}')
    print(f'pheno_data shapes are {pheno_data.train.shape}, {pheno_data.test.shape}')
    print(f'we have {X_train.shape[0]} samples and {X_train.shape[1]} total features, dtype is {X_train.dtype}')
    print()

    cov_number = min(X_train.shape[1] - fc, 1) # sex is always encoded
    model_path = get_model_path('main', run, fc, cfg.experiment.params.xgbpred, cov_number)
    print(f'model_path is {model_path}')
    eval_metric = ['auc', 'logloss'] if selector.is_clf else ['rmse']
    model, _, y_val_pred = train_booster_on_array(X_train, y_train, 
                                                   X_val, y_val, 
                                                   model_path, 
                                                   num_round=50,
                                                   tree_method='gpu_hist',
                                                   nthread=8,
                                                   base_score=y_train.mean(),
                                                   scale_pos_weight=1,
                                                   eval_metric=eval_metric,
                                                   deterministic_histogram=False,
                                                   is_clf=selector.is_clf,
                                                   **cfg.experiment.params.xgbpred)
    if not selector.is_clf:
        m = r2_score(y_val, y_val_pred)
        print(f'r2: {m:.4f} for {X_train.shape[1] - 10} total features')
    else:
        m = roc_auc_score(y_val, y_val_pred)
        print(f'mean of y_val_pred is {numpy.mean(y_val_pred):.5f}')
        print(f'auc: {m:.4f} for {X_train.shape[1] - 10} total features')

    pheno_data = load_phenotype_for_xgboost(cfg)
    limit = int(model.attributes()['best_iteration'])
    
    dmatrix = xgboost.DMatrix(X_test.values, missing=-9)
    
    _, y_pred = model.predict(dmatrix, ntree_limit=limit)

    y_pred = pandas.DataFrame(data=y_pred, columns=['y_pred'], index=X_test.index)

    print(f'We predicted {y_pred.shape[0]} rows from {X_test.shape} rows and features')
    print(f'columns of pheno_data are {pheno_data.test.columns}, y_pred columns are {y_pred.columns}')
    y = pheno_data.test.merge(y_pred, how='inner', left_index=True, right_index=True)
    print(X_test.head().index)
    print(pheno_data.test.head().index)    
    print(f'After merge with loaded test data with shape {pheno_data.test.shape} we have {y.shape[0]} rows to eval')
   
    if not selector.is_clf:
        m = r2_score(y.trait.values, y.y_pred.values)
        print(f'final test r2: {m:.4f} for {X_train.shape[1] - 10} total features')
    else:
        m = roc_auc_score(y.trait.values, y.y_pred.values)
        print(f'mean of y_test is {numpy.mean(y_pred):.5f}')
        print(f'auc: {m:.4f} for {X_train.shape[1] - 10} total features')



if __name__ == '__main__':
    run()