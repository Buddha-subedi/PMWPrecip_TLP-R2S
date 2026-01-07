# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:51:12 2024

@author: subed042
"""

def plot_confusion_matrix(y_test, y_test_pred, xlabel, ylabel, title):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm_ocean_im = confusion_matrix(y_test, y_test_pred)

    cm_tot_ocean_im = np.vstack([cm_ocean_im, np.sum(cm_ocean_im, axis=0)])
    cm_tot_ocean_im = np.hstack([cm_tot_ocean_im, np.sum(cm_tot_ocean_im, axis=1).reshape(-1, 1)])
    total_samples = np.sum(cm_ocean_im)  
    percent_equivalence = cm_ocean_im / total_samples * 100

    annotations = []

    for i in range(cm_tot_ocean_im.shape[0]):
        row_annotations = []
        for j in range(cm_tot_ocean_im.shape[1]):
            value = cm_tot_ocean_im[i, j]
            if i < cm_ocean_im.shape[0] and j < cm_ocean_im.shape[1]:
                percent_value = percent_equivalence[i, j]
                row_annotations.append(f"{value}\n({percent_value:.2f}%)")
            elif i == cm_ocean_im.shape[0] and j < cm_ocean_im.shape[1]:  # Total row
                actual_total = cm_tot_ocean_im[i, j]
                correct = cm_ocean_im[j, j]
                correct_percent = correct / actual_total * 100 if actual_total != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}% )\n({false_percent:.2f}%)")
            elif i < cm_ocean_im.shape[0] and j == cm_ocean_im.shape[1]:  # Total column
                predicted_total = cm_tot_ocean_im[i, j]
                correct = cm_ocean_im[i, i]
                correct_percent = correct / predicted_total * 100 if predicted_total != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}%)\n({false_percent:.2f}% )")
            else:  
                correct_total = np.trace(cm_ocean_im)
                correct_percent = correct_total / value * 100 if value != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}%)\n({false_percent:.2f}%)")
        annotations.append(row_annotations)


    annotations = np.array(annotations)

    mask = np.full(cm_tot_ocean_im.shape, '', dtype=object)

    for i in range(cm_ocean_im.shape[0]):
        mask[i, i] = 'diag'

    mask[-1, :] = 'total'
    mask[:, -1] = 'total'

    mask[mask == ''] = 'off_diag'

    colors = {'diag': 'lightblue', 'total': 'whitesmoke', 'off_diag': 'salmon'}
    cmap = sns.color_palette([colors[key] for key in ['diag', 'total', 'off_diag']])

    mask_num = np.zeros_like(mask, dtype=float)
    mask_num[mask == 'diag'] = 0
    mask_num[mask == 'total'] = 1
    mask_num[mask == 'off_diag'] = 2


    plt.figure(figsize=(4, 4))
    sns.heatmap(mask_num, annot=False, cmap=plt.cm.colors.ListedColormap(cmap),
                xticklabels=['noPrecip', 'Rain', 'Snow', 'Total'],
                yticklabels=['noPrecip', 'Rain', 'Snow', 'Total'],
                linewidths=1, linecolor='black', cbar=False)  

    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tick_params(axis='both', which='both', width=2, length=6)
    for i in range(cm_tot_ocean_im.shape[0]):
        for j in range(cm_tot_ocean_im.shape[1]):
            text = annotations[i, j]
            if i == cm_ocean_im.shape[0] or j == cm_ocean_im.shape[1]: 
                parts = text.split('\n')
                plt.text(j + 0.5, i + 0.45, parts[1],
                         ha='center', va='center', fontsize=8, fontweight='bold', rotation=0, color='blue')
                plt.text(j + 0.5, i + 0.65, parts[2],
                         ha='center', va='center', fontsize=8, fontweight='bold', rotation=0, color='salmon')
            else:
                plt.text(j + 0.5, i + 0.5, text,
                         ha='center', va='center', fontsize=8, fontweight='bold', rotation=45)

    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.show()
    



cdf_mapping = None
def learn_cdf_mapping(y, x):
    import scipy.stats as stats
    from scipy.interpolate import interp1d
    import pandas as pd
    import numpy as np
    global cdf_mapping

    y_series = pd.Series(y)
    x_lrn = pd.Series(x)
    y_series = y_series.clip(lower=x_lrn.min())

    cumfreq_pred = stats.cumfreq(y_series, numbins=min(len(y_series), 4000))
    Fi1 = cumfreq_pred.cumcount
    xi1 = cumfreq_pred.lowerlimit + np.arange(len(Fi1)) * cumfreq_pred.binsize
    Fi1 = Fi1 / max(Fi1)

    cumfreq_actual = stats.cumfreq(x_lrn, numbins=min(len(x_lrn), 4000))
    Fi2 = cumfreq_actual.cumcount
    xi2 = cumfreq_actual.lowerlimit + np.arange(len(Fi2)) * cumfreq_actual.binsize
    Fi2 = Fi2 / max(Fi2)  


    def enforce_increasing(x, f):
        x, idx = np.unique(x, return_index=True)  
        f = f[idx]
        if len(x) > 1 and np.any(np.diff(x) == 0):  
            x += np.linspace(1e-10, 1e-8, num=len(x))  
        return x, f

    xi1, Fi1 = enforce_increasing(xi1, Fi1)
    xi2, Fi2 = enforce_increasing(xi2, Fi2)

    if len(xi1) < 2 or len(xi2) < 2:
        raise ValueError("Interpolation failed: xi1 or xi2 has fewer than 2 unique values. Check input distributions.")

    # Store interpolation functions
    cdf_mapping = {
        "pred_to_cdf": interp1d(xi1, Fi1, kind='linear', fill_value="extrapolate", assume_sorted=True),
        "cdf_to_actual": interp1d(Fi2, xi2, kind='linear', fill_value="extrapolate", assume_sorted=True),
    }

def get_actual_from_prediction(y_pred_array):
    import numpy as np
    global cdf_mapping
    if cdf_mapping is None:
        raise ValueError("CDF mapping has not been learned yet. Call learn_cdf_mapping(y, x) first.")

    cdf_values = cdf_mapping["pred_to_cdf"](y_pred_array)

    actual_values = cdf_mapping["cdf_to_actual"](cdf_values)

    actual_values = np.where(np.isnan(actual_values), 0.0, actual_values)
    actual_values = np.where(actual_values < 0, 0.001, actual_values)

    return actual_values




cdf_mapping = None
def learn_cdf_mapping(y, x):
    import scipy.stats as stats
    from scipy.interpolate import interp1d
    import pandas as pd
    import numpy as np
    global cdf_mapping

    y_series = pd.Series(y)
    x_lrn = pd.Series(x)
    y_series = y_series.clip(lower=x_lrn.min())

    cumfreq_pred = stats.cumfreq(y_series, numbins=min(len(y_series), 4000))
    Fi1 = cumfreq_pred.cumcount
    xi1 = cumfreq_pred.lowerlimit + np.arange(len(Fi1)) * cumfreq_pred.binsize
    Fi1 = Fi1 / max(Fi1)

    cumfreq_actual = stats.cumfreq(x_lrn, numbins=min(len(x_lrn), 4000))
    Fi2 = cumfreq_actual.cumcount
    xi2 = cumfreq_actual.lowerlimit + np.arange(len(Fi2)) * cumfreq_actual.binsize
    Fi2 = Fi2 / max(Fi2)  


    def enforce_increasing(x, f):
        x, idx = np.unique(x, return_index=True)  
        f = f[idx]
        if len(x) > 1 and np.any(np.diff(x) == 0):  
            x += np.linspace(1e-10, 1e-8, num=len(x))  
        return x, f

    xi1, Fi1 = enforce_increasing(xi1, Fi1)
    xi2, Fi2 = enforce_increasing(xi2, Fi2)

    if len(xi1) < 2 or len(xi2) < 2:
        raise ValueError("Interpolation failed: xi1 or xi2 has fewer than 2 unique values. Check input distributions.")

    # Store interpolation functions
    cdf_mapping = {
        "pred_to_cdf": interp1d(xi1, Fi1, kind='linear', fill_value="extrapolate", assume_sorted=True),
        "cdf_to_actual": interp1d(Fi2, xi2, kind='linear', fill_value="extrapolate", assume_sorted=True),
    }

def get_actual_from_prediction(y_pred_array):
    import numpy as np
    global cdf_mapping
    if cdf_mapping is None:
        raise ValueError("CDF mapping has not been learned yet. Call learn_cdf_mapping(y, x) first.")

    cdf_values = cdf_mapping["pred_to_cdf"](y_pred_array)

    actual_values = cdf_mapping["cdf_to_actual"](cdf_values)

    actual_values = np.where(np.isnan(actual_values), 0.0, actual_values)
    actual_values = np.where(actual_values < 0, 0.001, actual_values)

    return actual_values




def TLPR2S_model(path_orbit, booster,  snow_rate_booster, rain_rate_booster, df_cdf_rain, df_cdf_snow):
    import xgboost as xgb
    import scipy.io
    import numpy as np
    import pandas as pd
    npz = np.load(path_orbit)
    flattened_dict = {key: npz[key].ravel() for key in npz.files}
    df = pd.DataFrame(flattened_dict)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    row_dimension, column_dimension = npz['10v'].shape
    latitude = npz['Latitude']
    longitude = npz['Longitude']



    input_vars = ['10v', '10h', '18v', '18h','23v','36v', '36h', '89v', '89h', '166v', '166h','183-3', '183-7',
              'tciw','tclw','tcwv','t2m','cape','u10', 'v10', 'skt',
              'sd', 'tcslw','tcw','swvl1','lsm', 'siconc']

    x_snow_rate = df_cdf_snow[input_vars]
    dtest_sr = xgb.DMatrix(x_snow_rate)
    y_snow_rate = snow_rate_booster.predict(dtest_sr)
    y_actual_snow_rate = df_cdf_snow['cdf_snow'].values

    x_rain_rate = df_cdf_rain[input_vars]
    dtest_rr = xgb.DMatrix(x_rain_rate)
    y_rain_rate = rain_rate_booster.predict(dtest_rr)
    y_actual_rain_rate = df_cdf_rain['cdf_rain'].values


    x = df[input_vars]
    d_x = xgb.DMatrix(x)
    y_phase_pred = booster.predict(d_x)


    df['pred_phase'] = y_phase_pred
    df['pred_snow_rate'] = 0
    df['pred_rain_rate'] = 0

    # Snow rate estimation
    df_snowr = df[df['pred_phase'] == 2]
    x_rate = df_snowr[input_vars]
    dtest_sr = xgb.DMatrix(x_rate)
    y_snow_rate_pred = snow_rate_booster.predict(dtest_sr)
    learn_cdf_mapping(y_snow_rate, y_actual_snow_rate)
    y_snow_rate_pred = np.clip(y_snow_rate_pred, np.min(y_snow_rate), np.max(y_snow_rate))
    actual_snow_values = get_actual_from_prediction(y_snow_rate_pred)
    df.loc[df_snowr.index, 'pred_snow_rate'] = actual_snow_values

    # Rain rate estimation
    df_rainr = df[df['pred_phase'] == 1]
    x_rain_rate = df_rainr[input_vars]
    dtest_rr = xgb.DMatrix(x_rain_rate)
    y_rain_rate_pred = rain_rate_booster.predict(dtest_rr)
    learn_cdf_mapping(y_rain_rate, y_actual_rain_rate)
    y_rain_rate_pred = np.clip(y_rain_rate_pred, np.min(y_rain_rate), np.max(y_rain_rate))
    actual_rain_values = get_actual_from_prediction(y_rain_rate_pred)
    df.loc[df_rainr.index, 'pred_rain_rate'] = actual_rain_values

    # Output preparation
    phase = np.empty((row_dimension, column_dimension), dtype='float64')
    rain = np.empty((row_dimension, column_dimension), dtype='float64')
    snow = np.empty((row_dimension, column_dimension), dtype='float64')

    k = 0
    for i in range(row_dimension):
        for j in range(column_dimension):
            phase[i, j] = df['pred_phase'].iloc[k]
            rain[i, j] = df['pred_rain_rate'].iloc[k]
            snow[i, j] = df['pred_snow_rate'].iloc[k]
            k += 1

    return phase, rain, snow, latitude, longitude  




def TLPR2S_model_mat(mat_file, booster, snow_rate_booster_tl, rain_rate_booster_tl, df_cdf_match_rain, df_cdf_match_snow):

    snowr_input = ['10v', '10h', '18v', '18h','23v','36v', '36h', '89v', '89h', '166v', '166h','183-3', '183-7',
              'tciw','tclw','tcwv','t2m','cape','u10', 'v10', 'skt',
              'sd', 'tcslw','tcw','swvl1','lsm', 'siconc']

    input_vars = ['10v', '10h', '18v', '18h','23v','36v', '36h', '89v', '89h', '166v', '166h','183-3', '183-7',
              'tciw','tclw','tcwv','t2m','cape','u10', 'v10', 'skt',
              'sd', 'tcslw','tcw','swvl1','lsm', 'siconc']

    x_snow_rate = df_cdf_match_snow[snowr_input]
    dtest_sr = xgb.DMatrix(x_snow_rate)
    y_snow_rate = snow_rate_booster_tl.predict(dtest_sr)
    y_actual_snow_rate = df_cdf_match_snow['cdf_snow'].values

    x_rain_rate = df_cdf_match_rain[snowr_input]
    dtest_rr = xgb.DMatrix(x_rain_rate)
    y_rain_rate = rain_rate_booster_tl.predict(dtest_rr)
    y_actual_rain_rate = df_cdf_match_rain['cdf_rain'].values

    print(f"Processing file: {mat_file}...")
    mat = scipy.io.loadmat(mat_file)
    X_data = mat['X'][0,0]
    a = X_data['ERA5_WVP']
    row_dimension, column_dimension = a.shape
    print('Row dimension is:')
    print(row_dimension)
    print('Column dimension is:')
    print(column_dimension)
    latitude= X_data['LatS2']
    longitude= X_data['LonS2']

    columns_and_arrays = {
        '10v': X_data['TbS1'][:, :, 0], '10h': X_data['TbS1'][:, :, 1],
        '18v': X_data['TbS1'][:, :, 2], '18h': X_data['TbS1'][:, :, 3],
        '23v': X_data['TbS1'][:, :, 4], '36v': X_data['TbS1'][:, :, 5],
        '36h': X_data['TbS1'][:, :, 6], '89v': X_data['TbS1'][:, :, 7],
        '89h': X_data['TbS1'][:, :, 8], '166v': X_data['TbS2'][:, :, 0],
        '166h': X_data['TbS2'][:, :, 1], '183-3': X_data['TbS2'][:, :, 2],
        '183-7': X_data['TbS2'][:, :, 3], 'tciw': X_data['ERA5_IWP'],
        'tclw': X_data['ERA5_LWP'], 'tcwv': X_data['ERA5_WVP'],
        't2m': X_data['ERA5_t2m'], 'cape': X_data['ERA5_CAPE'],
        'u10': X_data['ERA5_u10'], 'v10': X_data['ERA5_v10'],
        'cin': X_data['ERA5_cin'], 'skt': X_data['ERA5_skt'],
        'asn': X_data['ERA5_asn'], 'rsn': X_data['ERA5_rsn'],
        'sd': X_data['ERA5_sd'], 'tcslw': X_data['ERA5_tcslw'],
        'tcw': X_data['ERA5_tcw'], 'swvl1': X_data['ERA5_swvl1'],
        'lsm': X_data['ERA5_lsm'], 'siconc': X_data['ERA5_siconc'],
        'Latitude': X_data['LatS2'], 'Longitude': X_data['LonS2'],
        'Month': np.repeat(X_data['Month'], X_data['LatS2'].shape[0]),
        'Day': np.repeat(X_data['Day'], X_data['LatS2'].shape[0]),
        'mean_aspect': X_data['DEM_aspect_mean'], 
        'elevation_mean': X_data['DEM_elevation_mean']
    }

    df = pd.DataFrame({col: arr.flatten() for col, arr in columns_and_arrays.items()})


    x = df[input_vars]
    d_x = xgb.DMatrix(x)
    y_phase_pred = booster.predict(d_x)

    df['pred_phase'] = y_phase_pred
    df['pred_snow_rate'] = 0
    df['pred_rain_rate'] = 0

    # Snow rate estimation
    df_snowr = df[df['pred_phase'] == 2]
    x_rate = df_snowr[snowr_input]
    dtest_sr = xgb.DMatrix(x_rate)
    y_snow_rate_pred = snow_rate_booster_tl.predict(dtest_sr)
    learn_cdf_mapping(y_snow_rate, y_actual_snow_rate)
    y_snow_rate_pred = np.clip(y_snow_rate_pred, np.min(y_snow_rate), np.max(y_snow_rate))
    actual_snow_values = get_actual_from_prediction(y_snow_rate_pred)
    df.loc[df_snowr.index, 'pred_snow_rate'] = actual_snow_values

    # Rain rate estimation
    df_rainr = df[df['pred_phase'] == 1]
    x_rain_rate = df_rainr[snowr_input]
    dtest_rr = xgb.DMatrix(x_rain_rate)
    y_rain_rate_pred = rain_rate_booster_tl.predict(dtest_rr)
    learn_cdf_mapping(y_rain_rate, y_actual_rain_rate)
    y_rain_rate_pred = np.clip(y_rain_rate_pred, np.min(y_rain_rate), np.max(y_rain_rate))
    actual_rain_values = get_actual_from_prediction(y_rain_rate_pred)
    df.loc[df_rainr.index, 'pred_rain_rate'] = actual_rain_values

    # Output preparation
    phase = np.empty((row_dimension, column_dimension), dtype='float64')
    rain = np.empty((row_dimension, column_dimension), dtype='float64')
    snow = np.empty((row_dimension, column_dimension), dtype='float64')

    k = 0
    for i in range(row_dimension):
        for j in range(column_dimension):
            phase[i, j] = df['pred_phase'].iloc[k]
            rain[i, j] = df['pred_rain_rate'].iloc[k]
            snow[i, j] = df['pred_snow_rate'].iloc[k]
            k += 1

    return phase, rain, snow, latitude, longitude  





