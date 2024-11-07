import pandas as pd
import numpy as np
from glob import glob
import xgboost as xgb
from tqdm import tqdm
import os
import joblib
import re

def predict_with_xgboost_model(model, X_val):
    dval = xgb.DMatrix(X_val)
    return model.predict(dval)

Path = r"E:\ERA5DATA\val_pft_data\*.csv"
val_paths = glob(Path)

reco_model_paths = glob(r"E:\GPP\algo&products\RECO\RECO_MODEL\*.joblib")
reco_model_container = {}
for path in reco_model_paths:
    model = joblib.load(path)
    pft = int(re.findall(r"\d+", os.path.basename(path))[0])
    reco_model_container[pft] = model

nee_model_paths = glob(r"E:\GPP\algo&products\NEE\NEE_MODEL\*.joblib")
nee_model_container = {}
for path in nee_model_paths:
    model = joblib.load(path)
    pft = int(re.findall(r"\d+", os.path.basename(path))[0])
    nee_model_container[pft] = model

gpp_model_paths = glob(r"E:\GPP\algo&products\GPP\GPP_MODEL\*.joblib")
gpp_model_container = {}
for path in gpp_model_paths:
    model = joblib.load(path)
    pft = int(re.findall(r"\d+", os.path.basename(path))[0])
    gpp_model_container[pft] = model

output_dir = r'E:\GPP\algo&products\PRODUCTS'

for val_path in tqdm(val_paths):

    val_datas = pd.read_csv(val_path)

    if val_datas['GOSIF'].isnull().all():
        print(f"Skipping file {val_path} because GOSIF field is completely empty.")
        continue

    predict = {'time': [], 'longitude': [], 'latitude': [], 'predict_gpp': [], 'predict_nee': [], 'predict_reco': [], 'PFT': [], 'DEM': []}

    k = val_datas["time"][0].replace("-", "")[0:6]
    year_folder_path = os.path.join(output_dir, k[0:4])
    if not os.path.exists(year_folder_path):
        os.makedirs(year_folder_path)
    
    val_datas['month'] = int(k[4:6])

    # val_datas = val_datas[(val_datas['PFT'] != 15)]
    # val_datas = val_datas[(val_datas['PFT'] != 0)]
    antarctica_latitude_range = (-90, -60)
    val_datas = val_datas[
        ~(
            (val_datas['latitude'].between(*antarctica_latitude_range))
        )
    ]
    val_datas = val_datas.reset_index(drop=True)

    # Separate PFT 16 data and set predictions to zero
    pft_16_15_data = val_datas[val_datas['M_PFT'].isin([0, 13, 16])]
    val_datas = val_datas[~val_datas['M_PFT'].isin([0, 13, 16])]
    
    for i in range(len(pft_16_15_data)):
        predict['time'].append(k)
        predict['longitude'].append(pft_16_15_data['longitude'].iloc[i])
        predict['latitude'].append(pft_16_15_data['latitude'].iloc[i])
        predict['predict_gpp'].append(0)
        predict['predict_nee'].append(0)
        predict['predict_reco'].append(0)
        predict['PFT'].append(pft_16_15_data['M_PFT'].iloc[i])
        predict['DEM'].append(pft_16_15_data['DEM'].iloc[i])

    group_predict_features = val_datas.groupby('M_PFT')

    for pft, predict_feature in group_predict_features:
        predict_feature = predict_feature.reset_index(drop=True)
        val_features = np.array(predict_feature[['month','latitude','longitude','tp','lai_hv','d2m','t2m','ssrd','strd', 'sp','GOSIF']].values)

        try:
            predict_gpp = predict_with_xgboost_model(gpp_model_container[pft], val_features)
            predict_nee = predict_with_xgboost_model(nee_model_container[pft], val_features)
            predict_reco = predict_with_xgboost_model(reco_model_container[pft], val_features)
        except KeyError:
            print('No model for PFT no.', pft)
            continue

        for i in range(len(val_features)):
            predict['time'].append(k)
            predict['longitude'].append(predict_feature['longitude'][i])
            predict['latitude'].append(predict_feature['latitude'][i])
            predict['predict_gpp'].append(predict_gpp[i])
            predict['predict_nee'].append(predict_nee[i])
            predict['predict_reco'].append(predict_reco[i])
            predict['PFT'].append(pft)
            predict['DEM'].append(predict_feature['DEM'][i])

    df = pd.DataFrame(predict)
    output_file_path = os.path.join(year_folder_path, f'predict_{k}.csv')
    df.to_csv(output_file_path, index=False)
