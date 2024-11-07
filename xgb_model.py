import pandas as pd
import numpy as np
from glob import glob
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
import os
from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization
import joblib
import time
import datetime


def get_datas(paths):
    datas = pd.DataFrame()
    dic = {
        'WAT': 0, 'ENF': 1, 'EBF': 2, 'DNF': 3, 'DBF': 4, 'MF': 5, 'CSH': 6, 'OSH': 7,
        'WSA': 8, 'SAV': 9, 'GRA': 10, 'WET': 11, 'CRO': 12, 'URB': 13, 'CVM': 14, 'SNO': 15,
        'BSV': 16
    }
    for path in paths:
        data = pd.read_csv(path)
        data['month'] = data['TIMESTAMP'].astype(int) % 100
        data['diff_gpp'] = abs(data['GPP_NT_VUT_REF'] - data['GPP_DT_VUT_REF'])
        data = data[data['diff_gpp'] <= 1]
        data = data[['month', 'Lat', 'Lon', 'P_ERA', 'LAI', 'VPD_ERA', 'TA_ERA', 'SW_IN_ERA', 'LW_IN_ERA', 'PA_ERA', 'GOSIF', 'DEM', 'TIMESTAMP', 'NEE_VUT_REF_QC', 'PFT', 'Site', 'NEE_VUT_REF']]
        data['PFT'] = data['PFT'].map(dic)
        datas = pd.concat([datas, data])
    datas.replace(to_replace=-9999, value=np.nan, inplace=True)
    datas = datas[datas['NEE_VUT_REF_QC'] >= 0.75]
    datas = datas.drop_duplicates(subset=['Site', 'PFT','TIMESTAMP'])
    datas = datas.dropna()
    reversed_dic = {v: k for k, v in dic.items()} 
    
    return datas, reversed_dic

class XGBProblem(Problem):
    def __init__(self, group_data):
        group_data = np.array(group_data.values)
        self.X_train = group_data[:, 0:11]
        self.y_train = group_data[:, 16]
        super().__init__(dimension=6, lower=[3, 0.001, 0.5, 0.5, 0, 0], upper=[15, 0.1, 1, 1, 10, 10])

    def _evaluate(self, x):
        max_depth = int(x[0])
        eta = float(x[1])
        subsample = float(x[2])
        colsample_bytree = float(x[3])
        llambda = float(x[4])
        alpha = float(x[5])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_r2_scores = []
        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train[train_index], self.X_train[val_index]
            y_tr, y_val = self.y_train[train_index], self.y_train[val_index]
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': max_depth,
                'eta': eta,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'lambda': llambda,
                'alpha': alpha,
                'tree_method': 'hist',
                'grow_policy': 'lossguide',
                'booster': 'gbtree',
                'verbosity': 0,
                'n_jobs': -1 
            }
            train_data = xgb.DMatrix(X_tr, label=y_tr)
            val_data = xgb.DMatrix(X_val, label=y_val)
            evals = [(val_data, 'eval')]
            model = xgb.train(params, train_data, num_boost_round=10000, evals=evals, early_stopping_rounds=20, verbose_eval=False)
            y_pred = model.predict(val_data)
            rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
            r2 = metrics.r2_score(y_val, y_pred)
            rmse_r2_scores.append(0.4 * r2 - 0.6 * rmse)
        return -np.mean(rmse_r2_scores)


def train_xgb_model(pft, group_data, reversed_dic):

    problem = XGBProblem(group_data)
    task = Task(problem=problem, max_iters = 50)
    algo = ParticleSwarmOptimization(population_size = 10)
    best = algo.run(task)
    best_params = best[0]
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': int(best_params[0]),
        'eta': best_params[1],
        'subsample': best_params[2],
        'colsample_bytree': best_params[3],
        'lambda': best_params[4],
        'alpha': best_params[5],
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'booster': 'gbtree',
        'verbosity': 0
    }

    group_data = np.array(group_data.values)
    X_train = group_data[:, 0:11]
    y_train = group_data[:, 16]
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)

    evals = [(val_data, 'eval')]
    model = xgb.train(params, train_data, num_boost_round=10000, evals=evals, early_stopping_rounds=20, verbose_eval=False)

    print(reversed_dic.get(pft, 'Unknown PFT'))
    print("Best iteration:", model.best_iteration)
    print("Best score:", model.best_score)

    return model, best_params, model.best_iteration, model.best_score


PATH = r'E:\FLUXDATA\TrainDATA\*.csv'
all_paths = glob(PATH)

all_datas, reversed_dic = get_datas(all_paths)
group_datas = all_datas.groupby('PFT')


best_params_df = pd.DataFrame(columns=['PFT', 'max_depth', 'eta', 'subsample', 'colsample_bytree', 'lambda', 'alpha', 'best_iteration', 'best_score'])


for pft, group_data in group_datas:

    begin = time.time()
    bdt = datetime.datetime.fromtimestamp(begin)
    print("start:  ", bdt)

    Q1 = group_data['NEE_VUT_REF'].quantile(0.25)
    Q3 = group_data['NEE_VUT_REF'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group_data = group_data[(group_data['NEE_VUT_REF'] >= lower_bound) & (group_data['NEE_VUT_REF'] <= upper_bound)]

    print(reversed_dic.get(pft, 'Unknown PFT'), ':  ', len(group_data))

    model, best_params, best_iteration, best_score = train_xgb_model(pft, group_data, reversed_dic)

    param_row = {
        'PFT': reversed_dic.get(pft, 'Unknown PFT'),
        'max_depth': int(best_params[0]),
        'eta': best_params[1],
        'subsample': best_params[2],
        'colsample_bytree': best_params[3],
        'lambda': best_params[4],
        'alpha': best_params[5],
        'best_iteration': best_iteration,
        'best_score': best_score
    }

    best_params_df = pd.concat([best_params_df, pd.DataFrame([param_row])], ignore_index=True)
    joblib.dump(model, os.path.join(r'E:\GPP\algo&products\NEE\NEE_MODEL', f'NEE_{pft}.joblib'))

    end = time.time()
    edt = datetime.datetime.fromtimestamp(end)
    print("end:  ",edt)
    print("times:  ",end - begin)

best_params_df.to_csv(r"E:\GPP\algo&products\NEE\NEE_MODEL\best_params.csv", index=False)
