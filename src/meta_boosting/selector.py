import xgboost
import numpy
from tqdm import tnrange, tqdm
import pandas
import time
from pgenlib import PgenReader
import gc
from sklearn.preprocessing import StandardScaler
from .xgb_utils import pgen_mask_and_trait, sort_pgen_test_data, get_arrays_new, sort_phenotype_features
import pickle

import xgboost
import time


def select_features_narray(X, y, 
                           X_test, y_test,
                           X_valid, y_valid,
                           rounds=100, 
                           verbose=False,
                           is_clf=False,
                           history=None,
                           y_start=None,
                           y_test_start=None,
                           early_stopping_rounds=20,
                           **kwargs):
    
    start = time.time()
    #print(X.shape)
    # X[X == -9] = -1
    # X_test[X_test == -9] = -1
    # X_valid[X_valid == -9] = -1
    # print((X==0).sum()/X.size, (X==1).sum()/X.size, (X==2).sum()/X.size, (X==-1).sum()/X.size)
    dtrain = xgboost.DMatrix(X, label=y_start, nthread=1, missing=2)
    dtest = xgboost.DMatrix(X_test, label=y_test_start, nthread=1, missing=2)
    dvalid = xgboost.DMatrix(X_valid, nthread=1, missing=2)
    prevalence = (y_start.sum() / y_start.shape[0])
    kwargs['base_score'] = prevalence
    if y is not None:
        # here y and y_test are already margins
        dtrain.set_base_margin(y)
        dtest.set_base_margin(y_test)
        dvalid.set_base_margin(y_valid)
    
    print(f'creating xgboost DMatrix took {time.time() - start:.4f}, x dtype is {X.dtype}')

    #print('dmatrices loaded in narray')
    max_depth = 2
    objective = 'binary:logistic' if is_clf else 'reg:squarederror'
    eval_metric = ['auc', 'logloss'] if is_clf else ['rmse']
    
    param = {'objective': objective, 
             'verbosity': 0, 
             'eta': 0.1,
             'eval_metric': eval_metric,
             'max_depth': max_depth, 
             'colsample_bytree' : 1.0,
             'subsample': 0.25,
             'gamma': 0,
             'nthread': 8,
             'tree_method': 'gpu_hist',
             'gpu_id' : 0,
             'deterministic_histogram': False}
    
    for arg, value in kwargs.items():
        param[arg] = value
        
    evals = [(dtrain, 'train')]
    if X_test is not None:
        evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}
    verbose_eval = 20 if verbose else None
    bst = xgboost.train(param, dtrain, rounds, 
                        evals=evals, early_stopping_rounds=early_stopping_rounds, 
                        evals_result=evals_result, verbose_eval=verbose_eval)
    
    history['test'].append(min(evals_result['test'][eval_metric[0]]))
    trees = bst.trees_to_dataframe()
    trees_gain = trees[trees.Feature != 'Leaf'].Gain
    history['gain'].append(trees_gain)
    #history['trees'].append(trees)
    #history['model'].append(bst)
    print(f'we built {len(trees_gain)} nodes')
    
    y_train_pred = bst.predict(dtrain, 
                               output_margin=True,
                               ntree_limit=bst.best_ntree_limit)

    history['y_train_pred'].append(y_train_pred)
    
    y_train_pred = y_train_pred.reshape(-1, 1)
    del dtrain
    
    y_test_pred = bst.predict(dtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
    y_test_pred = y_test_pred.reshape(-1, 1)
    history['y_test_pred'].append(y_test_pred)
    del dtest
    
    y_valid_pred = bst.predict(dvalid, output_margin=True, ntree_limit=bst.best_ntree_limit)
    y_valid_pred = y_valid_pred.reshape(-1, 1)
    history['y_valid_pred'].append(y_valid_pred)
    del dvalid

    gain = bst.get_score(importance_type='total_gain')
    del bst
    return gain, y_train_pred, y_test_pred, y_valid_pred


class XGBSelector():
    def __init__(self,
                 pgen_train_dir,
                 pgen_test_dir,
                 pgen_valid_dir,
                 train_data,
                 test_data,
                 run,
                 feature_data=None,
                 snp_window_len=5000,
                 window_trees=1000, 
                 verbose=True,
                 m_eta=1.0,
                 reverse_direction=True,
                 is_clf=True,
                 prefix='les_',
                 selector_func=select_features_narray,
                 shuffle_snps=False,
                 early_stopping_rounds=20,
                 **booster_kwargs):
        
        self.pgen_train_dir = pgen_train_dir
        self.pgen_test_dir = pgen_test_dir
        self.pgen_valid_dir = pgen_valid_dir
        self.run = run
        self.chromosomes = run.chromosomes
        self.snp_window_len = snp_window_len
        self.window_trees = window_trees
        self.verbose = verbose
        self.m_eta = m_eta
        self.reverse_direction = reverse_direction
        self.is_clf = is_clf
        self.booster_kwargs = booster_kwargs
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_data = feature_data
        self.test_data = test_data
        self.train_data = train_data
        self.shuffle_snps = shuffle_snps
        
        self.first_chrom_path = f'{self.pgen_train_dir}/{prefix}chr{self.chromosomes[0]}'
        self.first_test_chrom_path = f'{self.pgen_test_dir}/{prefix}chr{self.chromosomes[0]}'
        self.first_valid_chrom_path = f'{self.pgen_valid_dir}/{prefix}chr{self.chromosomes[0]}'

        self.gain_data = {}
        
        self.files = [f'{prefix}chr{i}.pgen' for i in self.chromosomes]
        self.history = [{'test': [],
                         'gain': [],
                         'trees': [],
                         'y_train_pred': [],
                         'y_test_pred': [],
                         'y_valid_pred': [],
                         'model': []} for i in self.chromosomes]
        
        if self.feature_data is not None:
            self.history.append({'test': [],
                                 'gain': [],
                                 'trees': [],
                                 'y_train_pred': [],
                                 'y_test_pred': [],
                                 'y_valid_pred': [],
                                 'model': []})
        
        y_train, y_test = pgen_mask_and_trait(self.first_chrom_path, self.first_test_chrom_path, train_data)
        if not is_clf:
            self.scaler = StandardScaler()
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1))
            y_test = self.scaler.transform(y_test.reshape(-1, 1))
        
        self.selector_func = selector_func
        self.mask = ~numpy.isnan(y_train)
        self.test_mask = ~numpy.isnan(y_test)
        self.y_train_start = y_train[self.mask].astype(numpy.float).reshape(-1, 1)
        self.y_test_start = y_test[self.test_mask].astype(numpy.float).reshape(-1, 1)
        self.y_train = None
        self.y_test = None
        self.y_valid = None
        self.gain_data = {}
    
    def _load_range_data(self, reader, test_reader, valid_reader, sample_count, test_sample_count, valid_sample_count, features_count, left):
        X = numpy.zeros((sample_count, features_count), dtype=numpy.int8)
        X_test = numpy.zeros((test_sample_count, features_count), dtype=numpy.int8)
        X_valid = numpy.zeros((valid_sample_count, features_count), dtype=numpy.int8)
            
        if self.reverse_direction:
            reader.read_range(left - features_count, left, X, sample_maj=True, allele_idx=0)
            test_reader.read_range(left - features_count, left, X_test, sample_maj=True, allele_idx=0)
            valid_reader.read_range(left - features_count, left, X_valid, sample_maj=True, allele_idx=0)
        else:
            reader.read_range(left, left + features_count, X, sample_maj=True, allele_idx=0)
            test_reader.read_range(left, left + features_count, X_test, sample_maj=True, allele_idx=0)
            valid_reader.read_range(left, left + features_count, X_valid, sample_maj=True, allele_idx=0)
        
        return X.astype(numpy.float32), X_test.astype(numpy.float32), X_valid.astype(numpy.float32)
        
    def _load_shuffled_data(self, reader, test_reader, valid_reader, sample_count, test_sample_count, valid_sample_count, features_count, lefts, total_variant_count):
        Xs, Xts, Xvs = [], [], []
        for left in lefts:
            small_window_count = min(features_count // 10, total_variant_count - left)
                
            X, X_test, X_valid = self._load_range_data(
                reader, test_reader, valid_reader, sample_count, test_sample_count, valid_sample_count, small_window_count, left
            )
            Xs.append(X)
            Xts.append(X_test)
            Xvs.append(X_valid)
        
        return numpy.hstack(Xs), numpy.hstack(Xts), numpy.hstack(Xvs)
        
    def select_from_chrom(self, pgen_path, pgen_test_path, pgen_valid_path, chrom, history=None):
        
        reader = PgenReader(bytes(pgen_path, encoding='utf-8'))

        sample_count = reader.get_raw_sample_ct()
        variant_count = reader.get_variant_ct()
        chrom_gain_data = numpy.zeros((variant_count))

        test_reader = PgenReader(bytes(pgen_test_path, encoding='utf-8'))
        test_sample_count = test_reader.get_raw_sample_ct()

        valid_reader = PgenReader(bytes(pgen_valid_path, encoding='utf-8'))
        valid_sample_count = valid_reader.get_raw_sample_ct()

        iter_data = numpy.arange(0, variant_count, self.snp_window_len)
        order = numpy.arange(0, variant_count, self.snp_window_len // 10)
        if self.shuffle_snps:
            numpy.random.shuffle(order)
        elif self.reverse_direction:
            iter_data = numpy.arange(variant_count, -1, -self.snp_window_len)
            
        iterator = tqdm(iter_data)
        
        for i, left in enumerate(iterator):
                
            remainder = variant_count - left if not self.reverse_direction else left
            features_count = self.snp_window_len if remainder >= self.snp_window_len else remainder
            
            if self.shuffle_snps:
                lefts = sorted(order[i*10:(i+1)*10])
                features_count = min(variant_count - lefts[-1], self.snp_window_len, len(lefts)*(self.snp_window_len // 10))
                print('features_count and variant_count', features_count, variant_count)
                X, X_test, X_valid = self._load_shuffled_data(
                    reader, test_reader, valid_reader, 
                    sample_count, test_sample_count, valid_sample_count, 
                    self.snp_window_len, lefts, variant_count
                )
                gc.collect()
            else:
                start = time.time()
                X, X_test, X_valid = self._load_range_data(reader,
                                                           test_reader,
                                                           valid_reader,
                                                           sample_count, 
                                                           test_sample_count, 
                                                           valid_sample_count, 
                                                           features_count, 
                                                           left)
                end = time.time()
                print(f'load_range_data elapsed {end - start:.3f}')
            
            start = time.time()
            X_submit = sort_pgen_test_data(self.first_test_chrom_path, 
                                           self.first_valid_chrom_path, 
                                           X_test, X_valid, self.test_data)
            
            X, X_test, X_submit = self._add_features(X, X_test, X_submit)
            if self.y_valid_frame is not None:
                X_submit.loc[:, 'y'] = self.y_valid_frame['y']
                self.y_valid = X_submit.loc[:, 'y'].values.reshape(-1, 1)
                self.y_valid_frame = None
                del X_submit['y']
            end = time.time()
            print(f'sorting pgen test data and adding features elapsed {end - start:.4f}')
            # print('features added', X.shape, X_test.shape, X_submit.shape, self.y_train.shape, self.y_test.shape, self.y_valid.shape)
            start = time.time()
            block_gain, y_train_pred, y_test_pred, y_valid_pred = self.selector_func(
                                                                       X[self.mask.ravel()], 
                                                                       self.y_train,
                                                                       X_test[self.test_mask.ravel()],
                                                                       self.y_test,
                                                                       X_submit.values, self.y_valid,
                                                                       rounds=self.window_trees, 
                                                                       verbose=self.verbose,
                                                                       history=history,
                                                                       is_clf=self.is_clf,
                                                                       y_start=self.y_train_start,
                                                                       y_test_start=self.y_test_start,
                                                                       early_stopping_rounds=self.early_stopping_rounds, 
                                                                       **self.booster_kwargs)
            end = time.time()
            print(f'selector_func elapsed {end - start:.4f}')
            self.y_train = y_train_pred 
            self.y_test = y_test_pred 
            self.y_valid = y_valid_pred
            gc.collect()
            for feature, gain in block_gain.items():
                fi = int(feature[1:])
                if fi >= features_count: # additional features, i.e. sex, age...
                    continue
                if self.shuffle_snps:
                    lefts = sorted(order[i*10:(i+1)*10])
                    window_fi = lefts[fi // (self.snp_window_len // 10)]
                    chrom_gain_data[window_fi + fi % (self.snp_window_len // 10)] = gain
                elif self.reverse_direction:
                    chrom_gain_data[left - features_count + fi] = gain
                else:
                    chrom_gain_data[left + fi] = gain
                    
            self.gain_data[f'chr{chrom}'] = chrom_gain_data
            numpy.savez(self.run.results_path('selection_temp.npz'), **self.gain_data)
            
            with open(self.run.results_path('selector.pkl'), 'wb') as history_file:
                pickle.dump(self, history_file)
            
            #mem = tracker.SummaryTracker()
            #memory = pandas.DataFrame(mem.create_summary(), columns=['object', 'number_of_objects', 'memory'])
            #memory['mem_per_object'] = memory['memory'] / memory['number_of_objects']
            #print(memory.sort_values('memory', ascending=False).head(10))
            #print(memory.sort_values('mem_per_object', ascending=False).head(10))
        
        reader.close()
        test_reader.close()
        valid_reader.close()
        
        return 

    def _add_features(self, X_train, X_test, X_submit):
        return X_train, X_test, X_submit
        if self.feature_data is not None:
            train_features, test_features, valid_features = sort_phenotype_features(self.first_chrom_path,
                                                                                    self.first_test_chrom_path,
                                                                                    self.first_valid_chrom_path,
                                                                                    self.feature_data)

            return numpy.concatenate([X_train, train_features], axis=1), \
                   numpy.concatenate([X_test, test_features], axis=1), \
                   X_submit.merge(valid_features, how='inner', left_index=True, right_index=True)
        
        else:
            return X_train, X_test, X_submit
    
    def fit(self, start_index=0):
        # valid_features possibly not in the right order!!!
        if self.feature_data is not None:
            train_features, test_features, valid_features = sort_phenotype_features(self.first_chrom_path,
                                                                                    self.first_test_chrom_path,
                                                                                    self.first_valid_chrom_path,
                                                                                    self.feature_data)
            
            print(f'y_train_start: {self.y_train_start.shape}, {numpy.isnan(self.y_train_start).sum()}')
            _, y_train_pred, y_test_pred, y_valid_pred = self.selector_func(
                                                           train_features[self.mask.ravel()], 
                                                           self.y_train,
                                                           test_features[self.test_mask.ravel()],
                                                           self.y_test,
                                                           valid_features.values, None,
                                                           rounds=400,
                                                           verbose=self.verbose,
                                                           history=self.history[-1],
                                                           is_clf=self.is_clf,
                                                           y_start=self.y_train_start,
                                                           y_test_start=self.y_test_start,
                                                           **self.booster_kwargs)
            
            print('after fitting for additional covariates first')
            print(valid_features.shape, y_valid_pred.shape)
            self.y_train = y_train_pred 
            self.y_test = y_test_pred 
            self.y_valid_frame = pandas.DataFrame(data=y_valid_pred, index=valid_features.index, columns=['y'])
            
            
        if self.reverse_direction:
            iterator = tnrange(len(self.files)-1, -1, -1)
        else:
            # TODO: temporary costyl
            iterator = tnrange(start_index, len(self.files))
        
        for i in iterator:
            train_chrom_path = f'{self.pgen_train_dir}/{self.files[i]}'
            test_chrom_path = f'{self.pgen_test_dir}/{self.files[i]}'
            valid_chrom_path = f'{self.pgen_valid_dir}/{self.files[i]}'
            chrom = self.chromosomes[i]

            self.select_from_chrom(train_chrom_path, 
                                   test_chrom_path,
                                   valid_chrom_path,
                                   chrom,
                                   history=self.history[i])

        return self.gain_data
    
    def predict(self):
        y_train_pred = numpy.exp(self.history[-1]['y_train_pred'][-1])
        y_test_pred = numpy.exp(self.history[-1]['y_test_pred'][-1])
        y_valid_pred = numpy.exp(self.history[-1]['y_valid_pred'][-1])
        
        valid_frame = pandas.DataFrame(index=self.valid_index, data=y_valid_pred, columns=[self.test_data.columns[-1]])
        return y_train_pred, y_test_pred, valid_frame
    
    def transform(self, feature_count, prefix='les_', add_features=False, features=None):
        
        if add_features and self.feature_data is not None:
            feature_data = self.feature_data
        else:
            feature_data = features
            
        X_train, X_test, X_submit, y_train, y_test = get_arrays_new(
            self.train_data,
            self.test_data,
            self.run.results_path('selection_temp.npz'),
            self.pgen_train_dir,
            self.pgen_test_dir,
            self.pgen_valid_dir,
            self.chromosomes,
            is_clf=self.is_clf,
            snp_count=feature_count,
            feature_data=feature_data,
            prefix=prefix[:-1],
        )
        
        return X_train, X_test, X_submit, y_train, y_test
