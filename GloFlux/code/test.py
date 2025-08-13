import pandas as pd
import numpy as np
from glob import glob
import xgboost as xgb
import warnings
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
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
            df = df[['month', 'month_sin', 'month_cos', 'P_F', 'LAI', 'VPD_F', 'TA_F', 'SW_IN_F', 'PA_F', 'GOSIF',
                     'TIMESTAMP', 'NEE_VUT_REF_QC', 'PFT', 'Site', 'GPP_NT_VUT_REF']]
            df['PFT'] = df['PFT'].map(PFT_DIC)
            df = df[df['NEE_VUT_REF_QC'] >= 0.75]
            data = pd.concat([data, df])
        except Exception:
            continue

    data.replace(to_replace=-9999, value=np.nan, inplace=True)
    data = data.drop_duplicates(subset=['Site', 'PFT', 'TIMESTAMP']).dropna()
    return data, REVERSED_DIC

def pso_objective(params, X_train, y_train, X_val, y_val):
    max_depth = int(round(params[0]))
    eta, subsample, colsample_bytree = params[1], params[2], params[3]
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    val_dmatrix = xgb.DMatrix(X_val, label=y_val)
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'tree_method': 'hist',
        'verbosity': 0
    }
    model = xgb.train(param, train_dmatrix, num_boost_round=1000,
                      evals=[(val_dmatrix, 'eval')],
                      early_stopping_rounds=20,
                      verbose_eval=False)
    return model.best_score

def pso_optimize(X_train, y_train, n_particles=20, max_iter=30):
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    lb = [3, 0.001, 0.5, 0.5]
    ub = [15, 0.1, 1.0, 1.0]
    def obj(params):
        return pso_objective(params, X_train_opt, y_train_opt, X_val, y_val)

    best_params, _ = pso(obj, lb, ub, swarmsize=n_particles, maxiter=max_iter)
    return {
        'max_depth': int(round(best_params[0])),
        'eta': best_params[1],
        'subsample': best_params[2],
        'colsample_bytree': best_params[3]
    }

def train_xgb_model(train_data):
    X = train_data.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
    y = train_data.iloc[:, 14].values
    best_params = pso_optimize(X, y, n_particles=20, max_iter=30)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': best_params['max_depth'],
        'eta': best_params['eta'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'tree_method': 'hist',
        'verbosity': 0
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=3000,
                      evals=[(dval, 'eval')],
                      early_stopping_rounds=30,
                      verbose_eval=False)
    return model, best_params


def finetune_pft_model(base_model, train_group, best_params):
    X_train = train_group[:, [0,1,2,3,4,5,6,7,8,9]]
    y_train = train_group[:, 14]
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': best_params['max_depth'],
        'eta': best_params['eta'] * 0.5,
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'tree_method': 'hist',
        'verbosity': 0
    }
    dtrain = xgb.DMatrix(X_train_opt, label=y_train_opt)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, 'eval')],
                      early_stopping_rounds=10,
                      verbose_eval=False,
                      xgb_model=base_model)
    return model

TRAIN_PATH = r'E:\yqw_data\FLUXDATA\TrainDATA\train\*.csv'
train_paths = glob(TRAIN_PATH)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results_base = []
fold_results_pft = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_paths)):
    train_files = [train_paths[i] for i in train_idx]
    val_files = [train_paths[i] for i in val_idx]

    train_data, reversed_dic = get_datas(train_files)
    val_data, _ = get_datas(val_files)

    model, best_params = train_xgb_model(train_data)

    fold_rmse_r2_base = pd.DataFrame(columns=['Fold', 'PFT', 'R2', 'RMSE', 'Sample_Size'])
    fold_rmse_r2_pft = pd.DataFrame(columns=['Fold', 'PFT', 'R2', 'RMSE', 'Sample_Size'])

    all_y_val, all_y_pred_base, all_y_pred_pft = [], [], []

    group_datas = val_data.groupby('PFT')
    for pft, val_group in group_datas:
        pft_name = reversed_dic.get(pft, 'Unknown PFT')

        try:
            if not val_group.empty:
                Q1 = val_group['GPP_NT_VUT_REF'].quantile(0.25)
                Q3 = val_group['GPP_NT_VUT_REF'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                val_group = val_group[(val_group['GPP_NT_VUT_REF'] >= lower_bound) & (val_group['GPP_NT_VUT_REF'] <= upper_bound)]

            sample_size = len(val_group)
            if sample_size == 0:
                continue

            X_val = val_group.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
            y_val = val_group.iloc[:, 14].values
            dval = xgb.DMatrix(X_val, label=y_val)

            y_pred_base = model.predict(dval)
            rmse_base = np.sqrt(metrics.mean_squared_error(y_val, y_pred_base))
            r2_base = r2_score(y_val, y_pred_base)

            fold_rmse_r2_base = pd.concat([fold_rmse_r2_base, pd.DataFrame([{
                'Fold': fold + 1, 'PFT': pft_name,
                'R2': r2_base, 'RMSE': rmse_base, 'Sample_Size': sample_size
            }])], ignore_index=True)

            train_group = train_data[train_data['PFT'] == pft].values
            pft_model = finetune_pft_model(model, train_group, best_params)

            y_pred_pft = pft_model.predict(dval)
            rmse_pft = np.sqrt(metrics.mean_squared_error(y_val, y_pred_pft))
            r2_pft = r2_score(y_val, y_pred_pft)

            fold_rmse_r2_pft = pd.concat([fold_rmse_r2_pft, pd.DataFrame([{
                'Fold': fold + 1, 'PFT': pft_name,
                'R2': r2_pft, 'RMSE': rmse_pft, 'Sample_Size': sample_size
            }])], ignore_index=True)

            all_y_val.extend(y_val.tolist())
            all_y_pred_base.extend(y_pred_base.tolist())
            all_y_pred_pft.extend(y_pred_pft.tolist())

        except Exception:
            continue

    overall_r2_base = r2_score(all_y_val, all_y_pred_base)
    overall_rmse_base = np.sqrt(metrics.mean_squared_error(all_y_val, all_y_pred_base))
    overall_r2_pft = r2_score(all_y_val, all_y_pred_pft)
    overall_rmse_pft = np.sqrt(metrics.mean_squared_error(all_y_val, all_y_pred_pft))

    fold_rmse_r2_base = pd.concat([fold_rmse_r2_base, pd.DataFrame([{
        'Fold': fold + 1, 'PFT': 'Overall', 'R2': overall_r2_base,
        'RMSE': overall_rmse_base, 'Sample_Size': len(all_y_val)
    }])], ignore_index=True)

    fold_rmse_r2_pft = pd.concat([fold_rmse_r2_pft, pd.DataFrame([{
        'Fold': fold + 1, 'PFT': 'Overall', 'R2': overall_r2_pft,
        'RMSE': overall_rmse_pft, 'Sample_Size': len(all_y_val)
    }])], ignore_index=True)

    fold_results_base.append(fold_rmse_r2_base)
    fold_results_pft.append(fold_rmse_r2_pft)

all_folds_base = pd.concat(fold_results_base, ignore_index=True)
all_folds_pft = pd.concat(fold_results_pft, ignore_index=True)

summary_base = all_folds_base[all_folds_base['PFT'] != 'Overall'].groupby('PFT').agg({
    'R2': ['mean', 'std'],
    'RMSE': ['mean', 'std']
}).reset_index()
summary_base.columns = ['PFT', 'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std']

summary_pft = all_folds_pft[all_folds_pft['PFT'] != 'Overall'].groupby('PFT').agg({
    'R2': ['mean', 'std'],
    'RMSE': ['mean', 'std']
}).reset_index()
summary_pft.columns = ['PFT', 'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std']

all_folds_base.to_csv(r'E:\yqw_data\GPP\test_finetune\zzzGPP_cv_rmse_r2_basemodel1.csv', index=False)
all_folds_pft.to_csv(r'E:\yqw_data\GPP\test_finetune\zzzGPP_cv_rmse_r2_finetuned_pftmodel1.csv', index=False)
summary_base.to_csv(r'E:\yqw_data\GPP\test_finetune\zzzGPP_cv_summary_basemodel1.csv', index=False)
summary_pft.to_csv(r'E:\yqw_data\GPP\test_finetune\zzzGPP_cv_summary_finetuned_pftmodel1.csv', index=False)
