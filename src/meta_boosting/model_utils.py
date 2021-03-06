from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy
from omegaconf import DictConfig, OmegaConf
import pandas
import xgboost
import lightgbm
from catboost import CatBoostRegressor, Pool, metrics, FeaturesData
from sklearn.preprocessing import StandardScaler

from .xgb_utils import train_booster_on_array


@dataclass
class Data:
    X_train: numpy.ndarray
    X_val: numpy.ndarray
    X_test: pandas.DataFrame
    y_train: numpy.ndarray
    y_val: numpy.ndarray
    y_test: numpy.ndarray
    cov_count: int

    def standardize(self):
        scaler = StandardScaler()
        y_train = scaler.fit_transform(self.y_train.reshape(-1, 1))
        y_val = scaler.transform(self.y_val.reshape(-1, 1))
        if self.y_test is not None:
            y_test = scaler.transform(self.y_test.reshape(-1, 1))
        else:
            y_test = None
        
        data = Data(self.X_train, self.X_val, self.X_test, y_train, y_val, y_test, self.cov_count)
        data.scaler = scaler
        return data


class Model(ABC):

    def __init__(self, model_args: DictConfig) -> None:
        self.model_args = model_args

    @abstractmethod
    def fit(self, data: Data) -> numpy.ndarray:
        pass

    @abstractmethod
    def predict(self, data: Data) -> pandas.DataFrame:
        pass


class XGBoostModel(Model):

    def fit(self, data: Data) -> numpy.ndarray:
        eval_metric = ['auc', 'logloss'] if self.model_args.is_clf else ['rmse']
        model, _, y_val_pred = train_booster_on_array(data.X_train, data.y_train, 
                                                      data.X_val, data.y_val, 
                                                      self.model_args.model_path, 
                                                      base_score=data.y_train.mean(),
                                                      scale_pos_weight=1,
                                                      num_round=self.model_args.num_round,
                                                      eval_metric=eval_metric,
                                                      is_clf=self.model_args.is_clf,
                                                      **self.model_args.params)
        self.model = model
        return y_val_pred

    def load_model_from_path(self):
        booster = xgboost.Booster()
        booster.load_model(self.model_args.model_path)
        self.model = booster

    def predict(self, data: Data) -> Tuple[numpy.ndarray, pandas.DataFrame]:
        limit = int(self.model.attributes()['best_iteration'])
    
        dmatrix = xgboost.DMatrix(data.X_test.values, missing=-9)
        y_pred = self.model.predict(dmatrix, ntree_limit=limit)
        
        y_pred = pandas.DataFrame(data=y_pred, columns=['y_pred'], index=data.X_test.index)
        return y_pred


class LightGBMModel(Model):
    def fit(self, data: Data) -> numpy.ndarray:
        
        snp_count = data.X_train.shape[1] - data.cov_count
        snp_features = [f'snp_{i}' for i in range(snp_count)]
        cov_features =  [f'cov_{j}' for j in range(data.cov_count)]
        train = lightgbm.Dataset(data.X_train, 
                                 label=data.y_train, 
                                 feature_name=snp_features + cov_features, 
                                 categorical_feature=snp_features)
        train.set_init_score(numpy.zeros(data.y_train.shape) + data.y_train.mean())
        val = lightgbm.Dataset(data.X_val, 
                               label=data.y_val, 
                               feature_name=snp_features + cov_features, 
                               categorical_feature=snp_features)
        
        bst = lightgbm.train(OmegaConf.to_container(self.model_args.params), 
                             train, self.model_args.num_round, 
                             valid_sets=[train, val],
                             callbacks=[lightgbm.early_stopping(stopping_rounds=1000), lightgbm.log_evaluation(period=10)])

        bst.save_model(self.model_args.model_path)
        self.model = bst

        y_val_pred = bst.predict(data.X_val, num_iteration=bst.best_iteration)
        return y_val_pred

    def predict(self, data: Data) -> pandas.DataFrame:
        y_test_pred = self.model.predict(data.X_test.values, num_iteration=self.model.best_iteration)
        return y_test_pred


class CatBoostModel(Model):
    def fit(self, data: Data) -> numpy.ndarray:
        snp_count = data.X_train.shape[1] - data.cov_count
        
        model = CatBoostRegressor(
            early_stopping_rounds=100,
            loss_function='RMSE'
        )
        '''
        train_data = FeaturesData(
            cat_feature_data=data.X_train[:, :snp_count].astype(numpy.dtype('U8')).astype(object),
            num_feature_data=data.X_train[:, snp_count:]
        )
        val_data = FeaturesData(
            cat_feature_data=data.X_val[:, :snp_count].astype(numpy.dtype('U8')).astype(object),
            num_feature_data=data.X_val[:, snp_count:]
        )
        '''

        train_pool = Pool(data.X_train, data.y_train)
        val_pool = Pool(data.X_val, data.y_val)

        model.fit(train_pool,
                  eval_set=val_pool,
                  use_best_model=True)

        self.model = model
        y_val_pred = model.predict(val_pool)
        return y_val_pred

    def predict(self, data: Data) -> pandas.DataFrame:
        snp_count = data.X_train.shape[1] - data.cov_count

        '''
        test_data = FeaturesData(
            cat_feature_data=data.X_test.values[:, :snp_count].astype(numpy.dtype('U8')).astype(object),
            num_feature_data=data.X_test.values[:, snp_count:]
        )
        '''
        test_pool = Pool(data.X_test.values, data.y_test)
        y_test_pred = self.model.predict(test_pool)
        return y_test_pred

