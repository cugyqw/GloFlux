import pandas as pd
import numpy as np
from glob import glob
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import os
from sklearn.linear_model import LinearRegression
from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization


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
    datas = datas.drop_duplicates(subset=['Site', 'PFT','TIMESTAMP'])  # 删除重复行
    datas = datas.dropna()
    reversed_dic = {v: k for k, v in dic.items()}
    
    return datas, reversed_dic

class XGBProblem(Problem):
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        super().__init__(dimension=4, lower=[3, 0.001, 0.5, 0.5], upper=[15, 0.1, 1, 1])

    def _evaluate(self, x):
        max_depth = int(x[0])
        eta = float(x[1])
        subsample = float(x[2])
        colsample_bytree = float(x[3])
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
                'tree_method': 'hist',
                'grow_policy': 'lossguide',
                'booster': 'gbtree',
                'verbosity': 0
            }
            train_data = xgb.DMatrix(X_tr, label=y_tr)
            val_data = xgb.DMatrix(X_val, label=y_val)
            evals = [(val_data, 'eval')]
            model = xgb.train(params, train_data, num_boost_round=10000, evals=evals, early_stopping_rounds=20, verbose_eval=False)
            y_pred = model.predict(val_data)
            rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
            r2 = metrics.r2_score(y_val, y_pred)
            rmse_r2_scores.append(0.4*r2 - 0.6*rmse)
        return -np.mean(rmse_r2_scores)

class CustomCallback(xgb.callback.TrainingCallback):
    def __init__(self, eval_data, eval_labels, reversed_dic, pft):
        self.eval_data = eval_data
        self.eval_labels = eval_labels
        self.rmse_list = []
        self.r2_list = []
        self.reversed_dic = reversed_dic
        self.pft = pft
    
    def after_iteration(self, model, epoch, evals_log):
        predictions = model.predict(xgb.DMatrix(self.eval_data))
        rmse = np.sqrt(metrics.mean_squared_error(self.eval_labels, predictions))
        r2 = metrics.r2_score(self.eval_labels, predictions)
        self.rmse_list.append(rmse)
        self.r2_list.append(r2)
        # print(f"Epoch {epoch + 1}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
        return False

    def plot_metrics(self):
        epochs = range(1, len(self.rmse_list) + 1)
        plt.figure(figsize=(10, 6))

        plt.plot(epochs, self.rmse_list)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('RMSE Over Epochs')
        plt.savefig(os.path.join(r"E:\GPP\algo_comparsion\xgboost\NEE_TEST\train", f"rmse_line_{reversed_dic.get(pft, 'Unknown PFT')}.png"))
        plt.close()

        plt.plot(epochs, self.r2_list)
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.title('R² Over Epochs')
        plt.savefig(os.path.join(r"E:\GPP\algo_comparsion\xgboost\NEE_TEST\train", f"r2_line_{reversed_dic.get(pft, 'Unknown PFT')}.png"))
        plt.close()

        df_metrics = pd.DataFrame({
            'xgb_Epoch': range(1, len(self.rmse_list) + 1),
            'xgb_RMSE': self.rmse_list,
            'xgb_R²': self.r2_list
        })
        df_metrics.to_csv(os.path.join(r"E:\GPP\algo_comparsion\xgboost\NEE_TEST\train", f"metrics_{self.reversed_dic.get(self.pft, 'Unknown PFT')}.csv"), index=False)

def train_xgb_model(pft, group_data, reversed_dic):

    site_groups = group_data.groupby('Site')
    largest_site = max(site_groups, key=lambda x: len(x[1]))
    remaining_sites = [site for site in site_groups if site[0] != largest_site[0]]
    second_largest_site = max(remaining_sites, key=lambda x: len(x[1]))
    second_largest_site_data = second_largest_site[1]
    other_sites_data = group_data[~group_data['Site'].isin([second_largest_site[0]])]

    second_largest_site_data = np.array(second_largest_site_data.values)
    other_sites_data = np.array(other_sites_data.values)
        
    X_train = other_sites_data[:, 0:11]
    y_train = other_sites_data[:, 16]

    x_val_site_data = second_largest_site_data[:, 0:11]
    y_val_site_data = second_largest_site_data[:, 16]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.4, random_state=42)

    print(f'PFT {pft}: 训练集大小 = {X_train.shape}, 测试集大小 = {X_test.shape}, 验证集大小 = {x_val_site_data.shape}')

    problem = XGBProblem(X_train, y_train)
    task = Task(problem=problem, max_iters=50)
    algo = ParticleSwarmOptimization(population_size=10)
    best = algo.run(task)
    best_params = best[0]
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': int(best_params[0]),
        'eta': best_params[1],
        'subsample': best_params[2],
        'colsample_bytree': best_params[3],
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'booster': 'gbtree',
        'verbosity': 0
    }

    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)

    evals = [(val_data, 'eval')]
    custom_callback = CustomCallback(X_val, y_val, reversed_dic, pft)
    model = xgb.train(params, train_data, num_boost_round=10000, evals=evals, early_stopping_rounds=20, verbose_eval=False,
                      callbacks=[custom_callback])
    
    X_test = xgb.DMatrix(X_test)
    y_pred = model.predict(X_test)

    test_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_test': y_test
    })
    test_df.to_csv(os.path.join(r'E:\GPP\algo_comparsion\xgboost\NEE_TEST\test',f"{reversed_dic.get(pft, 'Unknown PFT')}.csv"), index=False)

    rmse, r2 = draw_r2_rmse(y_test, y_pred, reversed_dic, pft, site_name = 'null',
                        rmse_path= r'E:\GPP\algo_comparsion\xgboost\NEE_TEST\test',
                        r2_path=r'E:\GPP\algo_comparsion\xgboost\NEE_TEST\test',
                        type = 'test')
    
    custom_callback.plot_metrics()
    
    return model, x_val_site_data, y_val_site_data, second_largest_site[0], rmse, r2, best_params
def draw_r2_rmse(y_test, y_pred, reversed_dic, pft, site_name, rmse_path, r2_path, type):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, linestyle='-', marker='o',  linewidth=1, label='True')
    plt.plot(range(len(y_test)), y_pred, linestyle='-', marker='x',  linewidth=1, label='Predict')
    if type == 'val':
        plt.title(f"{reversed_dic.get(pft, 'Unknown PFT')} - {site_name}")
    else:
        plt.title(f"{reversed_dic.get(pft, 'Unknown PFT')}")
    plt.grid(True)
    plt.xlabel('number')
    plt.ylabel('value')
    plt.legend(bbox_to_anchor=(1, 1), fontsize='small', frameon=False)
    plt.savefig(os.path.join(rmse_path, f"{reversed_dic.get(pft, 'Unknown PFT')}_diff.png"))
    plt.close()

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, s=20)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.text(0.05, 0.95, f'$R^2 = {r2:.2f}$\n$RMSE = {rmse:.2f}$', 
            ha='left', va='center', transform=plt.gca().transAxes, fontsize=10)
    if type == 'val':
        plt.title(f"{reversed_dic.get(pft, 'Unknown PFT')} - {site_name}")
    else:
        plt.title(f"{reversed_dic.get(pft, 'Unknown PFT')}")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.xlim(min_val-1, max_val+1)
    plt.ylim(min_val-1, max_val+1)
    plt.plot([min_val-1, max_val+1], [min_val-1, max_val+1], 'r--')

    model = LinearRegression()
    x_reshaped = y_test.reshape(-1, 1)
    model.fit(x_reshaped, y_pred)
    y_lin_pred = model.predict(x_reshaped)
    plt.plot(y_test, y_lin_pred, color='g')
    plt.savefig(os.path.join(r2_path, f"{reversed_dic.get(pft, 'Unknown PFT')}_r2.png"))
    plt.close()

    return rmse, r2

PATH = r'E:\FLUXDATA\TrainDATA\*.csv'
all_paths = glob(PATH)
all_datas, reversed_dic = get_datas(all_paths)

group_datas = all_datas.groupby('PFT')

print('数据准备好了')

val_rmse_r2 = pd.DataFrame(columns=['PFT', 'SITE', 'R2', 'RMSE'])
test_rmse_r2 = pd.DataFrame(columns=['PFT', 'R2', 'RMSE'])
best_params = pd.DataFrame(columns=['PFT', 'max_depth', 'eta', 'subsample', 'colsample_bytree'])

for pft, group_data in group_datas:
    Q1 = group_data['NEE_VUT_REF'].quantile(0.25)
    Q3 = group_data['NEE_VUT_REF'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group_data = group_data[(group_data['NEE_VUT_REF'] >= lower_bound) & (group_data['NEE_VUT_REF'] <= upper_bound)]
    print(reversed_dic.get(pft, 'Unknown PFT'), ':  ', len(group_data))

    try:
    # if 1:
        model, X_test, y_test, site_name, test_rmse, test_r2, best_param = train_xgb_model(pft, group_data, reversed_dic)

        X_test_data = xgb.DMatrix(X_test)
        y_pred = model.predict(X_test_data)

        val_rmse, val_r2 = draw_r2_rmse(y_test, y_pred, reversed_dic, pft, site_name, 
                            rmse_path= r'E:\GPP\algo_comparsion\xgboost\NEE_TEST\val',
                            r2_path=r'E:\GPP\algo_comparsion\xgboost\NEE_TEST\val',
                            type = 'val')
        
        val_df = pd.DataFrame({
            'y_pred': y_pred,
            'y_test': y_test
        })
        val_df.to_csv(os.path.join(r'E:\GPP\algo_comparsion\xgboost\NEE_TEST\val',f"{reversed_dic.get(pft, 'Unknown PFT')}.csv"), index=False)
        
        val_row = {'PFT': reversed_dic.get(pft, 'Unknown PFT'), 'SITE': site_name, 'R2': val_r2, 'RMSE': val_rmse}
        test_row = {'PFT': reversed_dic.get(pft, 'Unknown PFT'), 'R2': test_r2, 'RMSE': test_rmse}
        param_row = {'PFT': reversed_dic.get(pft, 'Unknown PFT'), 'max_depth': int(best_param[0]), 'eta': best_param[1], 'subsample': best_param[2], 'colsample_bytree': best_param[3]}

        val_rmse_r2 = pd.concat([val_rmse_r2,pd.DataFrame([val_row])], ignore_index=True)
        test_rmse_r2 = pd.concat([test_rmse_r2,pd.DataFrame([test_row])], ignore_index=True)
        best_params = pd.concat([best_params,pd.DataFrame([param_row])], ignore_index=True)
        
    except Exception as e:
        print(f"Error processing PFT {pft}: {e}")

val_rmse_r2.to_csv(r"E:\GPP\algo_comparsion\xgboost\NEE_TEST\val\val_rmse_r2.csv", index=False)
test_rmse_r2.to_csv(r"E:\GPP\algo_comparsion\xgboost\NEE_TEST\test\test_rmse_r2.csv", index=False)
best_params.to_csv(r"E:\GPP\algo_comparsion\xgboost\NEE_TEST\train\best_params.csv", index=False)
