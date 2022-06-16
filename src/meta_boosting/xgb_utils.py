import pickle
import xgboost
import numpy
from tqdm import tnrange
import pandas
import subprocess
import os
from pgenlib import PgenReader
import gc
from sklearn.preprocessing import StandardScaler


def standardize(y_train, y_test):
    scaler = StandardScaler()
    s_train = scaler.fit_transform(y_train)
    s_test = scaler.transform(y_test)
    return scaler, s_train, s_test


def select_features_narray(X, y, 
                           X_test, y_test,
                           rounds=100, 
                           verbose=False,
                           is_clf=False,
                           history=None,
                           y_start=None,
                           y_test_start=None,
                           **kwargs):
    
    sample_weight = 3 if 'sample_weight' not in kwargs else kwargs['sample_weight']
    train_weight = [sample_weight if y else 1 for y in y_start]
    test_weight = [sample_weight if y else 1 for y in y_test_start]

    dtrain = xgboost.DMatrix(X, label=y_start, nthread=1, weight=train_weight, missing=-9)
    dtest = xgboost.DMatrix(X_test, label=y_test_start, nthread=1, weight=test_weight, missing=-9)
    prevalence = (y_start.sum() / y_start.shape[0])
    kwargs['base_score'] = prevalence
    if y is not None:
        # here y and y_test are already margins
        dtrain.set_base_margin(y)
        dtest.set_base_margin(y_test)
            
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
    
    if is_clf:
        scale_pos_weight = (y_start.shape[0] - y_start.sum())/y_start.sum()
        #print(f'scale_pos_weight is {scale_pos_weight}')
        param['scale_pos_weight'] = scale_pos_weight
    
    for arg, value in kwargs.items():
        param[arg] = value
        
    #print(param)
        
    evals = [(dtrain, 'train')]
    if X_test is not None:
        evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}
    verbose_eval = 20 if verbose else None
    bst = xgboost.train(param, dtrain, rounds, 
                        evals=evals, early_stopping_rounds=2, 
                        evals_result=evals_result, verbose_eval=verbose_eval)
    
    history['test'].append(min(evals_result['test'][eval_metric[0]]))
    trees = bst.trees_to_dataframe()
    trees_gain = trees[trees.Feature != 'Leaf'].Gain
    history['gain'].append(trees_gain)
    history['trees'].append(trees)
    history['model'].append(bst)
    print(f'we built {len(trees_gain)} nodes')
    
    y_train_pred = bst.predict(dtrain, 
                               output_margin=True,
                               ntree_limit=bst.best_ntree_limit)

    history['y_train_pred'].append(y_train_pred)
    
    y_train_pred = y_train_pred.reshape(-1, 1)
    del dtrain
    if X_test is not None:
        y_test_pred = bst.predict(dtest, 
                                  output_margin=True,
                                  ntree_limit=bst.best_ntree_limit)
        y_test_pred = y_test_pred.reshape(-1, 1)
        history['y_test_pred'].append(y_test_pred)
        
        del dtest
    else:
        y_test_pred = None
        
    #print(y_train_pred.shape, y_test_pred.shape)
    gain = bst.get_score(importance_type='total_gain')
    #bst.save_model(f'temp_small_model.model')
    #del bst
    return gain, y_train_pred, y_test_pred


def filter_snps_range(start, stop, source_path, dest_path):
    
    completed = subprocess.run(['plink2', '--bfile', source_path,  
                                '--from', start, '--to', stop, '--threads', '8',
                                '--make-bed', '--out', dest_path],
                                text=True, stdout=subprocess.PIPE)
    return completed.returncode


def select_features_pgen(pgen_path,
                         y_train_cache,
                         mask,
                         pgen_test_path=None,
                         y_test_cache=None,
                         test_mask=None,
                         snp_window_len=1000,
                         window_trees=100, 
                         m_eta=0.5,
                         verbose=False,
                         history=None,
                         reverse_direction=False,
                         is_clf=False,
                         y_train_start=None,
                         y_test_start=None,
                         selector_func=None,
                         max_iterations=None,
                         **kwargs):
    
    reader = PgenReader(bytes(pgen_path, encoding='utf-8'))
    
    sample_count = reader.get_raw_sample_ct()
    variant_count = reader.get_variant_ct()
    
    file_gain_data = numpy.zeros((variant_count))
    if pgen_test_path is not None:
        test_reader = PgenReader(bytes(pgen_test_path, encoding='utf-8'))
        test_sample_count = test_reader.get_raw_sample_ct()

    if reverse_direction:
        iterator = tnrange(variant_count-1, -1, -snp_window_len)
    else:
        iterator = tnrange(0, variant_count, snp_window_len)
    
    iters = 0
    for left in iterator:
        if max_iterations is not None:
            if iters >= max_iterations:
                break
            iters += 1
            
    #for left in tnrange(0, snp_window_len*5, snp_window_len):
        remainder = variant_count - left if not reverse_direction else left
        features_count = snp_window_len if remainder >= snp_window_len else remainder
        X = numpy.zeros((sample_count, features_count), dtype=numpy.int8)
        if reverse_direction:
            reader.read_range(left - features_count, left, X, sample_maj=True, allele_idx=0)
        else:
            reader.read_range(left, left + features_count, X, sample_maj=True, allele_idx=0)
        
        X_test = numpy.zeros((test_sample_count, features_count), dtype=numpy.int8)
            
        if reverse_direction:
            test_reader.read_range(left - features_count, left, X_test, sample_maj=True, allele_idx=0)
        else:
            test_reader.read_range(left, left + features_count, X_test, sample_maj=True, allele_idx=0)
            
        block_gain, y_train_pred, y_test_pred = selector_func(X[mask.ravel()], 
                                                              y_train_cache,
                                                              X_test[test_mask.ravel()],
                                                              y_test_cache,
                                                              rounds=window_trees, 
                                                              verbose=verbose,
                                                              history=history,
                                                              is_clf=is_clf,
                                                              y_start=y_train_start,
                                                              y_test_start=y_test_start,
                                                              **kwargs)
            
        y_test_cache = y_test_pred #if y_test_cache is None else y_test_cache + m_eta*y_test_pred
        y_train_cache = y_train_pred #if y_train_cache is None else y_train_cache + m_eta*y_train_pred
        # for shap values we get array instead of dict
        #file_gain_data[left: left + features_count] = block_gain
        gc.collect()
        for feature, gain in block_gain.items():
            if reverse_direction:
                file_gain_data[left - features_count + int(feature[1:])] = gain
            else:
                file_gain_data[left + int(feature[1:])] = gain
    
    return file_gain_data, y_train_cache, y_test_cache


def mask_and_reindex(dataset, chrom_path):
    psam = pandas.read_csv(f'{chrom_path}.psam', sep='\t', index_col='IID')
    fam = dataset.fam.loc[pandas.notnull(dataset.fam.trait), ['iid', 'trait']].set_index('iid')
    iids = set(fam.index)
    psam_mask = psam.index.isin(iids)
    intersection = psam[psam_mask].join(fam, how='inner')
    dataset.trait = intersection.trait
    return psam_mask
        
    
def pgen_mask_and_trait(pgen_train_path, pgen_test_path, trait_data, is_clf=True):
    psam_train = pandas.read_csv(pgen_train_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    
    psam_test = pandas.read_csv(pgen_test_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    if is_clf:
        # because binary phenotypes are stored as 1 and 2 instead of 0 and 1
        if min(trait_data.iloc[:, -1]) > 0:
            psam_train['trait'] = trait_data.iloc[:, -1] - 1
            psam_test['trait'] = trait_data.iloc[:, -1] - 1
        else:
            psam_train['trait'] = trait_data.iloc[:, -1]
            psam_test['trait'] = trait_data.iloc[:, -1]
    else:
        psam_train['trait'] = trait_data.iloc[:, -1]
        psam_test['trait'] = trait_data.iloc[:, -1]
    
    print(psam_train)
    print(trait_data)
    print(psam_train.dtypes)
    print(trait_data.dtypes)
    return psam_train['trait'].values, psam_test['trait'].values
    

def select_imputed_features_new(pgen_dir,
                                pgen_test_dir,
                                results_path,
                                chromosomes,
                                trait_data=None,
                                y_train_precomputed=None,
                                y_test_precomputed=None,
                                snp_window_len=1000,
                                window_trees=100, 
                                verbose=False,
                                m_eta=0.5,
                                reverse_direction=False,
                                is_clf=False,
                                selector_func=select_features_narray,
                                history=None,
                                max_iterations=None,
                                **kwargs):
    
    gain_data = {}
    files = [f'f_chr{i}.pgen' for i in chromosomes]
    if history is None:
        history = [{'test': [],
                    'gain': [],
                    'trees': [],
                    'y_train_pred': [],
                    'y_test_pred': [],
                    'model': []} for i in chromosomes]
    
    first_chrom_path = f'{pgen_dir}/f_chr{chromosomes[0]}'
    first_test_chrom_path = f'{pgen_test_dir}/f_chr{chromosomes[0]}'
        
    y_train, y_test = pgen_mask_and_trait(first_chrom_path, first_test_chrom_path, trait_data, is_clf=is_clf)
    if not is_clf:
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = scaler.transform(y_test.reshape(-1, 1))
            
    mask = ~numpy.isnan(y_train)
    test_mask = ~numpy.isnan(y_test)
    y_train_start = y_train[mask].astype(numpy.float).reshape(-1, 1)
    y_test_start = y_test[test_mask].astype(numpy.float).reshape(-1, 1)
    
    if y_train_precomputed is not None:
        y_train = y_train_precomputed
        y_test = y_test_precomputed
    else:
        y_train, y_test = None, None
        
    if reverse_direction:
        iterator = tnrange(len(files)-1, -1, -1)
    else:
        iterator = tnrange(len(files))
    
    for i in iterator:
        pgen_path = f'{pgen_dir}/{files[i]}'
        pgen_test_path = f'{pgen_test_dir}/{files[i]}'
            
        file_gain_data, y_train, y_test = select_features_pgen(pgen_path,
                                                               y_train,
                                                               mask,
                                                               pgen_test_path=pgen_test_path,
                                                               y_test_cache=y_test,
                                                               test_mask=test_mask,
                                                               snp_window_len=snp_window_len,
                                                               window_trees=window_trees, 
                                                               m_eta=m_eta,
                                                               verbose=verbose,
                                                               history=history[i],
                                                               reverse_direction=reverse_direction,
                                                               is_clf=is_clf,
                                                               y_train_start=y_train_start,
                                                               y_test_start=y_test_start,
                                                               selector_func=selector_func,
                                                               max_iterations=max_iterations,
                                                               **kwargs)
    
        gain_data[f'chr{chromosomes[i]}'] = file_gain_data
        numpy.savez(results_path, **gain_data)        
    
    return gain_data, history


def select_features(train_array_dir, 
                    results_path,
                    snp_window_len=1000,
                    window_trees=100, 
                    subsample=0.25,
                    eta=0.1,
                    verbose=False):
    
    gain_data = []
    y_train_cache = None
    y_train_pred = None

    file_left = 0
    files = [file for file in os.listdir(train_array_dir) if file.endswith('.npz')]
    
    files = sorted(files, key=lambda f: int(f.replace('.npz', '').replace('array', '')))
    for i in tnrange(len(files)):
        file = files[i]
            
        loaded_train = numpy.load(f'{train_array_dir}/{file}')
        X, y = loaded_train['X'], loaded_train['y']
        
        file_gain_data = numpy.zeros((X.shape[1]))
        
        for left in range(0, X.shape[1], snp_window_len):
            X_s = X[:, left: left + snp_window_len]
            
            if y_train_pred is not None:
                y_train_cache -= y_train_pred
            else:
                y_train_cache = y
                
            block_gain, y_train_pred = select_features_narray(X_s, y_train_cache,
                                                              rounds=window_trees, 
                                                              eta=eta,
                                                              subsample=subsample,
                                                              verbose=verbose)
            
            for feature, gain in block_gain.items():
                file_gain_data[left + int(feature[1:])] = gain
                
        del X
        gain_data.append(file_gain_data)
        numpy.save(results_path, numpy.hstack(gain_data))        
    
    return gain_data
    

def train_booster_on_array(X_train, y_train, X_test, y_test, model_name, num_round, verbose_eval=50, evals_result=None, is_clf=False, **bst_args):

    dtrain = xgboost.DMatrix(X_train, label=y_train, missing=-9)
    del X_train
    dtest = xgboost.DMatrix(X_test, label=y_test, missing=-9)
    del X_test
    
    gc.collect()
    
    prevalence = (y_train.sum() / y_train.shape[0])
    #print(f'prevalence is {prevalence:.5f}')
    return train_booster_on_dmatrix(dtrain, dtest, model_name, num_round, is_clf=is_clf, verbose_eval=verbose_eval, evals_result=evals_result, **bst_args)


def booster_predict_on_array(X, model_path):
    bst = xgboost.Booster()
    bst.load_model(model_path)
    print(bst.attributes())
    limit = int(bst.attributes()['best_iteration'])
    
    dmatrix = xgboost.DMatrix(X, missing=-9)
    
    y_pred = bst.predict(dmatrix)
    return y_pred


def train_booster_on_dmatrix(dtrain, dtest, model_path, num_round, is_clf=False, verbose_eval=50, evals_result=None, **bst_args):

    max_depth = 2
    objective = 'binary:logistic' if is_clf else 'reg:squarederror'
    eval_metric = ['logloss', 'auc'] if is_clf else 'rmse'
    
    param = {'objective': objective, 
             'eval_metric': eval_metric,
             'verbosity': 0, 
             'eta': 0.02,
             'max_depth': max_depth, 
             'colsample_bytree' : 0.5,
             'subsample': 0.75,
             'nthread': 8,
             'gpu_id': 0,
             'tree_method': 'gpu_hist'}
    
    if is_clf:
        y = dtrain.get_label()
        scale_pos_weight = (y.shape[0] - y.sum())/y.sum()
        print(f'scale_pos_weight is {scale_pos_weight}')
        param['scale_pos_weight'] = scale_pos_weight
        
    for key, arg in bst_args.items():
        param[key] = arg
    
    print(param)
    evals = [(dtrain, 'train'), (dtest, 'valid')]

    evals_result = {} if evals_result is None else evals_result
    print('starting training')
    bst = xgboost.train(param, dtrain, num_round, evals=evals, early_stopping_rounds=200, evals_result=evals_result, verbose_eval=verbose_eval)
    
    bst.save_model(model_path)

    # make prediction
    xgb_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    return bst, dtest.get_label(), xgb_pred 


def sort_phenotype_features(pgen_train_path, pgen_test_path, pgen_valid_path, feature_data):
    
    cols = feature_data.columns
    psam_train = pandas.read_csv(pgen_train_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    train_frame = pandas.DataFrame(data=None, index=psam_train.index, columns=cols)
    train_frame.loc[:, cols] = feature_data
    
    psam_test = pandas.read_csv(pgen_test_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    test_frame = pandas.DataFrame(data=None, index=psam_test.index, columns=cols)
    test_frame.loc[:, cols] = feature_data
    
    psam_valid = pandas.read_csv(pgen_valid_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    valid_frame = pandas.DataFrame(data=None, index=psam_valid.index, columns=cols)
    valid_frame.loc[:, cols] = feature_data
    
    return train_frame.values.astype(numpy.float32), test_frame.values.astype(numpy.float32), valid_frame


def sort_pgen_test_data(pgen_test_path, pgen_valid_path, X_test, X_valid, test_trait_data, test_mask=None, valid_mask=None):
    
    psam_test = pandas.read_csv(pgen_test_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    
    psam_valid = pandas.read_csv(pgen_valid_path+'.psam', sep='\t', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
        
    if test_mask is None:
        X_test_frame = pandas.DataFrame(data=X_test, index=psam_test.index)
        X_valid_frame = pandas.DataFrame(data=X_valid, index=psam_valid.index)
    else:
        X_test_frame = pandas.DataFrame(data=X_test, index=psam_test.index[test_mask])
        X_valid_frame = pandas.DataFrame(data=X_valid, index=psam_valid.index[valid_mask])
    
    X_frame = pandas.concat([X_test_frame, X_valid_frame])
    
    merged = test_trait_data.merge(X_frame, how='inner', left_index=True, right_index=True)
    return merged.iloc[:, -X_test.shape[1]:]


def get_arrays_new(trait_data,
                   test_trait_data,
                   results_path, 
                   common_path,
                   common_test_path,
                   common_valid_path,
                   chromosomes,
                   is_clf=False,
                   snp_count=1000,
                   feature_data=None,
                   snp_indices=None,
                   standardize=False,
                   prefix='f'):
    
    if prefix:
        prefix = prefix + '_'
        
    first_chrom_path = f'{common_path}/{prefix}chr{chromosomes[0]}'
    first_test_chrom_path = f'{common_test_path}/{prefix}chr{chromosomes[0]}'
    first_valid_chrom_path = f'{common_valid_path}/{prefix}chr{chromosomes[0]}'
    
    y_train, y_test = pgen_mask_and_trait(first_chrom_path, first_test_chrom_path, trait_data, is_clf=is_clf)
    if not is_clf and standardize:
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = scaler.transform(y_test.reshape(-1, 1))
            
    mask = ~numpy.isnan(y_train)
    test_mask = ~numpy.isnan(y_test)
        
    y_train = y_train[mask].astype(numpy.float).reshape(-1, 1)
    y_test = y_test[test_mask].astype(numpy.float).reshape(-1, 1)
    
    train_dataset = MultiPgenDataset(common_path, results_path, count=snp_count, indices=snp_indices, prefix=prefix)
    X_train = train_dataset.get_array()
    
    print(f'X_test was loaded from {common_test_path}')
    test_dataset = MultiPgenDataset(common_test_path, results_path, count=snp_count, indices=snp_indices, prefix=prefix)
    X_test = test_dataset.get_array()

    valid_dataset = MultiPgenDataset(common_valid_path, results_path, count=snp_count, indices=snp_indices, prefix=prefix)
    X_valid = valid_dataset.get_array()
    #print('masks: ', numpy.isnan(y_train).sum(), numpy.isnan(y_test).sum())

    X_train = X_train[mask.ravel()]
    
    X_submit = sort_pgen_test_data(first_test_chrom_path, first_valid_chrom_path, X_test, X_valid, test_trait_data)

    X_test = X_test[test_mask.ravel()]
    psam_path = f'{first_test_chrom_path}.psam'
    psam = pandas.read_csv(psam_path, sep='\t')
    print(f'first indices are: {psam.IID[test_mask.ravel()].values[:10]}')
    if feature_data is not None:
        # here features are sorted by psam
        train_features, test_features, valid_features = sort_phenotype_features(
                                                            first_chrom_path, 
                                                            first_test_chrom_path, 
                                                            first_valid_chrom_path,
                                                            feature_data)
        
        #print(X_test.shape, test_features.shape, test_features[test_mask].shape)
        #print(X_train.shape, train_features.shape, train_features[mask].shape)
        #numpy.concatenate([X_train, train_features[mask]], axis=1), 
        #numpy.concatenate([X_test, test_features[test_mask]], axis=1), 
        return numpy.concatenate([X_train, train_features[mask]], axis=1), \
               numpy.concatenate([X_test, test_features[test_mask]], axis=1), \
               X_submit.merge(valid_features, how='inner', left_index=True, right_index=True), \
               y_train, \
               y_test
    
    return X_train, X_test, X_submit, y_train, y_test


# for selecting features
class MultiPgenDataset:
    def __init__(self, pgen_dir, importance_data_path, count=1000, indices=None, prefix='f_'):
        self.pgen_dir = pgen_dir
        self.count = count
        self.masks = None
        
        if indices is not None:
            chromosomes = indices.keys()
            self.files = {chrom: f'{pgen_dir}/{prefix}chr{chrom}.pgen' for chrom in chromosomes}
            self.index = indices
        else:
            self.importance_data = numpy.load(importance_data_path)
            chromosomes = self.importance_data.keys()
            self.files = {chrom: f'{pgen_dir}/{prefix}{chrom}.pgen' for chrom in chromosomes}
            self.index = self._build_index()
            #self._build_x_mask()
        
    def _build_x_mask(self):
        iids = set()
        
        for chrom in self.importance_data.keys():
            psam_path = f'{self.pgen_dir}/f{chrom}.psam'
            psam = pandas.read_csv(psam_path, sep='\t')
            if len(iids) == 0:
                iids = set(psam.IID)
            else:
                iids &= set(psam.IID)
                
        self.masks = {}
        
        for chrom in self.importance_data.keys():
            psam_path = f'{self.pgen_dir}/f{chrom}.psam'
            psam = pandas.read_csv(psam_path, sep='\t')
            self.masks[chrom] = psam.IID.isin(iids)
            
        
    def _build_index(self):
        
        tuples = []
        for chrom, data in self.importance_data.items():
            tuples.extend([(chrom, value, index) \
                           for index, value in enumerate(data)])
        
        tuples = sorted(tuples, key=lambda x:-x[1])[:self.count]
        
        indices = {chrom: [] for chrom in self.importance_data.keys()}
        for chrom, value, index in tuples:
            indices[chrom].append(index)
        
        return indices
        
    def get_array(self):
        arrays = []
        
        for chrom, file in self.files.items():
            indices = numpy.array(self.index[chrom], dtype=numpy.uint32)
            # print(f'top10 indices are: {indices[:10]}')
            # print(f'indices sum is {sum(indices)}')

            reader = PgenReader(bytes(file, encoding='utf-8'))
            sample_count = reader.get_raw_sample_ct()
            
            array = numpy.zeros((sample_count, len(indices)), dtype=numpy.int8)
            reader.read_list(indices, array, sample_maj=True, allele_idx=0)
            print(chrom, array.shape)
            if self.masks is not None:
                mask = self.masks[chrom]
                arrays.append(array[mask])
            else:
                arrays.append(array)
            
        return numpy.concatenate(arrays, axis=1)
        
        
def get_train_test_indices(train_path, test_path, trait_data, is_clf=False, chrom='MAIN', prefix=''):
    
    first_chrom_path = f'{train_path}/{prefix}chr{chrom}'
    first_test_chrom_path = f'{test_path}/{prefix}chr{chrom}'
    
    y_train, y_test = pgen_mask_and_trait(first_chrom_path, first_test_chrom_path, trait_data, is_clf=is_clf)
            
    mask = ~numpy.isnan(y_train)
    test_mask = ~numpy.isnan(y_test)
    
    train_psam = pandas.read_table(first_chrom_path + '.psam', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    test_psam = pandas.read_table(first_test_chrom_path + '.psam', header=0, names=['FID', 'IID', 'SEX'], index_col=['FID', 'IID'])
    return train_psam.loc[mask.ravel()], test_psam.loc[test_mask.ravel()]


def load_selector(run):
    with open(run.results_path('selector.pkl'), 'rb') as file:
        selector = pickle.load(file)

    return selector

def get_model_path(prefix, run, fc, params, cov_number):
    raw_name = prefix + f'_{fc // 1000}k_cov{cov_number}_'
    names = []
    for name, value in params.items():
        if name != 'interaction_constraints':
            names.append(f'{name}-{value}')
        else:
            names.append('noint')
    raw_name += '_'.join(names) + '.model'
    return run.model_path(raw_name)