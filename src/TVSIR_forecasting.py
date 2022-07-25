#!/usr/bin/env python3
import os.path
from datetime import timedelta
from glob import glob
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd


def main():
    state_data_glob_path = os.path.join('.', 'Data', 'State_Data_Clean', '*')
    for cleaned_data_path in sorted(glob(state_data_glob_path)):
        state_name = os.path.basename(cleaned_data_path).split('_', 1)[0]
        predicted_data_path = os.path.join('.', 'predictions', state_name + "_model_predictions.csv")
        predict_tvsir(cleaned_data_path, predicted_data_path)
    #
#


def predict_tvsir(cleaned_data_path, predicted_data_path):
    output_data_path = predicted_data_path.replace('_model_predictions.csv', '_tvsir_predictions.csv')
    output_rmse_path = output_data_path.replace('.csv', '_rmse.csv')
    # load data
    df_cleaned = pd.read_csv(cleaned_data_path, index_col='date', parse_dates=['date'])
    df_predicted = pd.read_csv(predicted_data_path, index_col='date', parse_dates=['date'])
    # select columns
    sir_cols = [
        col for col in df_cleaned.columns
        if 'census_pop' in col or 'confirmed' in col or 'death' in col
    ]
    df_sir_truth = df_cleaned[sir_cols].copy()
    bg_cols = [
        col for col in df_predicted.columns
        if col.endswith('_ensemble')
    ]
    df_bg_predicted = df_predicted[bg_cols].copy()
    start_date = df_bg_predicted.index[0]
    df_tvsir_prediction = pd.DataFrame(index=df_bg_predicted.index.copy())
    df_tvsir_prediction_rmse_by_series = dict()
    print('start date: ', start_date)
    sources = ['cdc', 'jh']
    window_sizes = [7, 14]
    for source in sources:
        n_col = [col for col in df_sir_truth.columns if col.__contains__('census_pop')][0]
        i_col = [col for col in df_sir_truth.columns if col.endswith(source + '_confirmed_ratio')][0]
        r_col = [col for col in df_sir_truth.columns if col.endswith(source + '_death_ratio')][0]
        i_truth_col = [col for col in df_sir_truth.columns if col.endswith(source + '_confirmed')][0]
        r_truth_col = [col for col in df_sir_truth.columns if col.endswith(source + '_death')][0]
        for col in df_sir_truth.columns:
            if 'census_pop' in col or 'confirmed' in col or 'death' in col:
                df_tvsir_prediction[col] = df_sir_truth[df_sir_truth.index >= start_date][col]
            #
        #
        start_nir = df_sir_truth[df_sir_truth.index == start_date][[n_col, i_col, r_col]]
        total_pop = start_nir[n_col][0]
        i0 = start_nir[i_col][0]
        r0 = start_nir[r_col][0]
        s0 = 1.0 - i0 - r0
        init = (s0, i0, r0)
        for window_size in window_sizes:
            print('series: ', source, window_size)
            b_col = [col for col in df_bg_predicted.columns if col.__contains__(source + '_beta_' + str(window_size))][0]
            g_col = [col for col in df_bg_predicted.columns if col.__contains__(source + '_gamma_' + str(window_size))][0]
            bg_series = [tuple(row) for row in df_bg_predicted[[b_col, g_col]].values]
            tvsir_prediction_output = time_varying_sir_ode(init, bg_series)
            tvsir_predictions_by_date = {
                (start_date + timedelta(days=t)): [i, r, i * total_pop, r * total_pop]
                for t, (s, i, r) in enumerate(tvsir_prediction_output)
            }
            tvsir_prediction_columns = [
                i_col.replace('confirmed_ratio', "bg%s_cases_ratio_predicted" % window_size),
                r_col.replace('death_ratio', "bg%s_mortality_ratio_predicted" % window_size),
                i_col.replace('confirmed_ratio', "bg%s_cases_predicted" % window_size),
                r_col.replace('death_ratio', "bg%s_mortality_predicted" % window_size),
            ]
            for col_index, predicted_col_name in enumerate(tvsir_prediction_columns):
                df_tvsir_prediction[predicted_col_name] = df_tvsir_prediction.index.map(
                    lambda date: tvsir_predictions_by_date[date][col_index]
                )
            #
            i_predicted_col = tvsir_prediction_columns[2]
            r_predicted_col = tvsir_prediction_columns[3]
            case_rmse = sqrt(mean_squared_error(df_tvsir_prediction[i_truth_col], df_tvsir_prediction[i_predicted_col]))
            mort_rmse = sqrt(mean_squared_error(df_tvsir_prediction[r_truth_col], df_tvsir_prediction[r_predicted_col]))
            i_rmse_col = i_predicted_col.replace('_predicted', '')
            r_rmse_col = r_predicted_col.replace('_predicted', '')
            df_tvsir_prediction_rmse_by_series[i_rmse_col] = case_rmse
            df_tvsir_prediction_rmse_by_series[r_rmse_col] = mort_rmse
            print('- RMSE:  ', i_rmse_col, ': ', case_rmse, '  ;  ', r_rmse_col, ': ', mort_rmse)
        #
    #
    print('saving predictions to: ', output_data_path)
    df_tvsir_prediction.to_csv(output_data_path)
    print('saving rmse to: ', output_rmse_path)
    with open(output_rmse_path, 'w') as output_rmse_file:
        for key, value in df_tvsir_prediction_rmse_by_series.items():
            output_rmse_file.write("%s,%s\n" % (key, value))
        #
    #
#


# init = (s0, i0, r0)
# bg_series = [ (b[0], g[0]), (b[1], g[1]), ... , (b[t_max], g[t_max]) ]
def time_varying_sir_ode(init, bg_series):
    s0, i0, r0 = init
    if (s0 + i0 + r0 - 1.0) > 1e-8:
        raise ValueError("Initial values do not sum to 1.0: %s" % init)
    #
    sir_series = [init]
    for t, (bt, gt) in enumerate(bg_series):
        st, it, rt = sir_series[t]
        ds = -bt * st * it
        di = bt * st * it - gt * it
        dr = gt * it
        sir_next = (st + ds, it + di, rt + dr)
        sir_series.append(sir_next)
    #
    return sir_series
#


if __name__ == '__main__':
    main()
#

