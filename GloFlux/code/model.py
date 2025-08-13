import pandas as pd
import numpy as np
from glob import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from pyswarm import pso

PFT_DIC = {
    'WAT': 0, 'ENF': 1, 'EBF': 2, 'DNF': 3, 'DBF': 4, 'MF': 5, 'CSH': 6, 'OSH': 7,
    'WSA': 8, 'SAV': 9, 'GRA': 10, 'WET': 11, 'CRO': 12, 'URB': 13, 'CVM': 14, 'SNO': 15, 'BSV': 16
}
REVERSED_DIC = {v: k for k, v in PFT_DIC.items()}

def get_datas(train_paths):
    data = pd.DataFrame()
    for path in train_paths:
        try:
            df = pd.read_csv(path)
            df['month'] = df['TIMESTAMP'].astype(int) % 100
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df = df[['month', 'month_sin', 'month_cos', 'P_F', 'LAI', 'VPD_F', 'TA_F',
                     'SW_IN_F', 'PA_F', 'GOSIF', 'TIMESTAMP', 'NEE_VUT_REF_QC', 'PFT', 'Site', 'GPP_NT_VUT_REF']]
            df['PFT'] = df['PFT'].map(PFT_DIC)
            df = df[df['NEE_VUT_REF_QC'] >= 0.75]
            data = pd.concat([data, df])
        except:
            continue

    data.replace(-9999, np.nan, inplace=True)
    data = data.drop_duplicates(subset=['Site', 'PFT', 'TIMESTAMP']).dropna()
    return data, REVERSED_DIC

def iqr_filter(df, col='GPP_NT_VUT_REF'):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def pso_optimize(X_train, y_train, X_val, y_val):
    def objective(params):
        max_depth, eta, subsample, colsample_bytree = params
        model = xgb.train(
            {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': int(round(max_depth)),
                'eta': eta,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'tree_method': 'hist',
                'verbosity': 0
            },
            xgb.DMatrix(X_train, label=y_train),
            num_boost_round=1000,
            evals=[(xgb.DMatrix(X_val, label=y_val), 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        return model.best_score
    lb = [3, 0.001, 0.5, 0.5]
    ub = [15, 0.1, 1.0, 1.0]
    best_params, _ = pso(objective, lb, ub, swarmsize=20, maxiter=30)
    return {
        'max_depth': int(round(best_params[0])),
        'eta': best_params[1],
        'subsample': best_params[2],
        'colsample_bytree': best_params[3]
    }


def train_xgb_model(train_data):
    X_train = train_data.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
    y_train = train_data.iloc[:, 14].values

    X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    best_params = pso_optimize(X_train_opt, y_train_opt, X_val, y_val)
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'verbosity': 0
    })

    model = xgb.train(best_params, xgb.DMatrix(X_train_opt, label=y_train_opt),
                      num_boost_round=3000,
                      evals=[(xgb.DMatrix(X_val, label=y_val), 'eval')],
                      early_stopping_rounds=30, verbose_eval=False)
    return model, best_params


def finetune_pft_model(base_model, train_group, best_params, pft_name):
    X_train = train_group.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
    y_train = train_group.iloc[:, 14].values

    X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    params = best_params.copy()
    params['eta'] *= 0.5

    model = xgb.train(params, xgb.DMatrix(X_train_opt, label=y_train_opt),
                      num_boost_round=1000,
                      evals=[(xgb.DMatrix(X_val, label=y_val), 'eval')],
                      early_stopping_rounds=10, verbose_eval=False,
                      xgb_model=base_model)

    joblib.dump(model, rf'E:\yqw_data\GPP\model\gpp\GPP_{pft_name}_finetuned.joblib')

def main():
    TRAIN_PATH = r'E:\yqw_data\FLUXDATA\TrainDATA\train\*.csv'
    train_paths = glob(TRAIN_PATH)
    train_data, reversed_dic = get_datas(train_paths)

    base_model, best_params = train_xgb_model(train_data)

    group_datas = train_data.groupby('PFT')
    for pft, group in group_datas:
        pft_name = reversed_dic.get(pft, 'Unknown PFT')
        group = iqr_filter(group)
        if len(group) == 0:
            continue
        finetune_pft_model(base_model, group, best_params, pft_name)

if __name__ == "__main__":
    main()
