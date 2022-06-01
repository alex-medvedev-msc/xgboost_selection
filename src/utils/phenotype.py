from dataclasses import dataclass
import pandas
from omegaconf import DictConfig
from ukb_loader import BinarySDLoader, UKBDataLoader
import os


@dataclass
class PhenoData:
    train: pandas.DataFrame
    test: pandas.DataFrame
    features: pandas.DataFrame


@dataclass
class UKBPhenoData:
    train: pandas.DataFrame
    val: pandas.DataFrame
    test: pandas.DataFrame
    pheno_code: str


@dataclass
class PCData:
    train: pandas.DataFrame
    val: pandas.DataFrame
    test: pandas.DataFrame


def load_pca(pca_dir: str) -> PCData:
    train_path = os.path.join(pca_dir, 'train', 'projections.txt')
    val_path = os.path.join(pca_dir, 'val', 'projections.txt')
    test_path = os.path.join(pca_dir, 'test', 'projections.txt')

    train_pca = pandas.read_table(train_path)
    val_pca = pandas.read_table(val_path)
    test_pca = pandas.read_table(test_path)

    cols = ['IID'] + [f'PC{i}' for i in range(1, 11)] # 10 Principal Components
    
    return PCData(
        train_pca.loc[:, cols].set_index('IID'), 
        val_pca.loc[:, cols].set_index('IID'), 
        test_pca.loc[:, cols].set_index('IID')
    )


def add_pca_to_ukb_data(ukb_data: UKBPhenoData, pc_data: PCData) -> UKBPhenoData:
    new_cols = pc_data.train.columns.tolist() + ukb_data.train.columns.tolist()
    print(f'Before PCA joining train had {ukb_data.train.shape[0]} samples')
    print(f'we have {new_cols} features in pc+ukb phenotype data')
    train = ukb_data.train.merge(pc_data.train, how='inner', left_index=True, right_index=True)
    val = ukb_data.val.merge(pc_data.val, how='inner', left_index=True, right_index=True)
    test = ukb_data.test.merge(pc_data.test, how='inner', left_index=True, right_index=True)
    train, val, test = train[new_cols], val[new_cols], test[new_cols]
    return UKBPhenoData(train, val, test, ukb_data.pheno_code)


def _ukb_phenotype_to_xgboost(ukb_data: UKBPhenoData) -> PhenoData:
    xgb_train = pandas.concat([ukb_data.train, ukb_data.val], axis=0).iloc[:, -1:].rename({ukb_data.pheno_code: 'trait'}, axis='columns')
    xgb_train = xgb_train.set_index(xgb_train.index, append=True)
    xgb_train.index.set_names(['FID', 'IID'], inplace=True)

    xgb_test = ukb_data.test.iloc[:, -1:].rename({'20002': 'trait'}, axis='columns')
    xgb_test = xgb_test.set_index(xgb_test.index, append=True)
    xgb_test.index.set_names(['FID', 'IID'], inplace=True)

    xgb_features = pandas.concat([ukb_data.train, ukb_data.val, ukb_data.test], axis=0).iloc[:, :-1]
    xgb_features = xgb_features.set_index(xgb_features.index, append=True)
    xgb_features.index.set_names(['FID', 'IID'], inplace=True)
    return PhenoData(xgb_train, xgb_test, xgb_features)


def load_csv_phenotype(cfg: DictConfig) -> UKBPhenoData:
    code = '45' # ultrasound device id, dummy phenotype
    features = [str(f) for f in cfg.experiment.phenotype.features]
    loader = UKBDataLoader(cfg.dataset.split.location, cfg.dataset.split.name, code, features)
    train, val, test = loader.load_train(), loader.load_val(), loader.load_test()

    phenotype_path = cfg.experiment.phenotype.csv
    phenotype_data = pandas.read_csv(phenotype_path, index_col=0)
    print(train.index.dtype, phenotype_data.index.dtype)
    train = train.merge(phenotype_data, how='inner', left_index=True, right_index=True).drop(code, axis='columns')
    val = val.merge(phenotype_data, how='inner', left_index=True, right_index=True).drop(code, axis='columns')
    test = test.merge(phenotype_data, how='inner', left_index=True, right_index=True).drop(code, axis='columns')
    return UKBPhenoData(train, val, test, cfg.experiment.phenotype.code)


def load_sd_phenotype(cfg: DictConfig) -> UKBPhenoData:
    features = [str(f) for f in cfg.experiment.phenotype.features]
    loader = BinarySDLoader(cfg.dataset.split.location, cfg.dataset.split.name, '20002', features, cfg.experiment.phenotype.code, na_as_false=cfg.experiment.phenotype.na_as_false)
    train, val, test = loader.load_train(), loader.load_val(), loader.load_test()
    return UKBPhenoData(train, val, test, cfg.experiment.phenotype.code)


def load_real_phenotype(cfg: DictConfig) -> UKBPhenoData:
    array_agg_func = cfg.experiment.phenotype.get('array_agg_func', 'mean')
    # print(f'array aggregating function is {array_agg_func}')
    features = [str(f) for f in cfg.experiment.phenotype.features]
    loader = UKBDataLoader(cfg.dataset.split.location, cfg.dataset.split.name, str(cfg.experiment.phenotype.code), features, array_agg_func=array_agg_func)
    train, val, test = loader.load_train(), loader.load_val(), loader.load_test()
    return UKBPhenoData(train, val, test, cfg.experiment.phenotype.code) 


def load_phenotype_for_xgboost(cfg: DictConfig) -> PhenoData:
    pcs = load_pca(cfg.dataset.pca_dir)
    if 'source' in cfg.experiment.phenotype and cfg.experiment.phenotype.source == 'csv':
        return _ukb_phenotype_to_xgboost(add_pca_to_ukb_data(load_csv_phenotype(cfg), pcs))
    elif cfg.experiment.phenotype.type == 'sd':
        return _ukb_phenotype_to_xgboost(add_pca_to_ukb_data(load_sd_phenotype(cfg), pcs))
    elif cfg.experiment.phenotype.type == 'real':
        return _ukb_phenotype_to_xgboost(add_pca_to_ukb_data(load_real_phenotype(cfg), pcs))
    else:
        raise ValueError(f'unknown phenotype type: {cfg.experiment.phenotype.type}, should be one of ["csv", "sd", "real"]')