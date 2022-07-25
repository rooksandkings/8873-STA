# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from math import sqrt
import numpy as np
import pandas as pd
import csv


# clean list of US states. No US territories.
def get_state_list():
    google_df = pd.read_csv(".\\Data\\Mobility_Data\\Google_Data\\2021_US_Region_Mobility_Report.csv", index_col='date')

    states_us_list = google_df['sub_region_1'].unique()
    states_us_list = [state for state in states_us_list if pd.isnull(state) == False]  # remove null
    states_us_list = [state for state in states_us_list if state != 'District of Columbia']  # remove Washington DC
    return states_us_list

def ols(state_covid_clean, train_start_date, train_end_date, test_prediction_start_date,
        test_range, feature, predictions_df, period):
    X = state_covid_clean.drop(columns=[feature])
    y_truth = state_covid_clean[feature].shift(-period)

    X_train_df = X[train_start_date.strftime('%m-%d-%Y'):(train_end_date - timedelta(days=1)).strftime('%m-%d-%Y')]
    y_train_df = y_truth[train_start_date.strftime('%m-%d-%Y'):
                         (train_end_date - timedelta(days=1)).strftime('%m-%d-%Y')]

    X_train = X_train_df.values
    y_train = y_train_df.values

    truth = []
    OLS_predictions = []
    dates = []

    # walk-forward validation (prediction)
    for t in pd.date_range(test_prediction_start_date, test_prediction_start_date + timedelta(days=test_range)):
        ols_model = OLS(y_train, X_train)
        ols_res = ols_model.fit()
        X_new = (X.loc[t.strftime('%m-%d-%Y')]).values
        y_pred = ols_res.predict(X_new)
        new_pred_train = X[train_start_date.strftime('%m-%d-%Y'):(t - timedelta(days=1)).strftime('%m-%d-%Y')]
        X_train = new_pred_train.values
        truth.append(y_truth[t.strftime('%m-%d-%Y')])
        OLS_predictions.append(y_pred[0])
        y_train = np.append(y_train, y_pred[0])
        dates.append(t.strftime('%m-%d-%Y'))

    # evaluate forecasts
    OLS_rmse = sqrt(mean_squared_error(truth, OLS_predictions))
    model_name = 'OLS'
    predictions_df = prediction_aggregator(predictions_df, dates,
                                           OLS_predictions, truth, feature, model_name)
    return predictions_df, OLS_rmse, model_name


def d_tree(state_covid_clean, train_start_date, train_end_date, test_prediction_start_date,
           test_range, feature, predictions_df, period,DT=False, RT=False, Bagged_DT=False, Bagged_RT=False,
           Boostrap=False, n_estimators=10, max_depth=20):
    X = state_covid_clean.drop(columns=[feature])
    y_truth = state_covid_clean[feature]
    y_truth = state_covid_clean[feature].shift(-period)

    X_train_df = X[train_start_date.strftime('%m-%d-%Y'):(train_end_date - timedelta(days=1)).strftime('%m-%d-%Y')]
    y_train_df = y_truth[train_start_date.strftime('%m-%d-%Y'):
                         (train_end_date - timedelta(days=1)).strftime('%m-%d-%Y')]

    X_train = X_train_df.values
    y_train = y_train_df.values

    truth = []
    d_tree_predictions = []
    dates = []

    # walk-forward validation (prediction)
    for t in pd.date_range(test_prediction_start_date, test_prediction_start_date + timedelta(days=test_range)):
        if DT:
            model_reggressor = DecisionTreeRegressor(max_depth=max_depth)  # vanilla decision tree
            model_reggressor.fit(X_train, y_train)
            model_name = 'DT'
        elif RT:
            model_reggressor = DecisionTreeRegressor(max_depth=max_depth, splitter='random')  # random forest
            model_reggressor.fit(X_train, y_train)
            model_name = 'RF'
        elif Bagged_DT:
            model_reggressor = BaggingRegressor(base_estimator=DecisionTreeRegressor(
                max_depth=max_depth), n_estimators=n_estimators, bootstrap=True).fit(
                X_train, y_train)  # forest regressor fit
            model_name = 'Bagged_DT'
        elif Bagged_RT:
            model_reggressor = BaggingRegressor(base_estimator=DecisionTreeRegressor(
                max_depth=max_depth, splitter='random'), n_estimators=n_estimators, bootstrap=True).fit(
                X_train, y_train)  # random forest regressor fit
            model_name = 'Bagged_RT'

        X_new = (X.loc[t.strftime('%m-%d-%Y')]).values
        y_pred = model_reggressor.predict(X_new.reshape(1, -1))
        new_pred_train = X[train_start_date.strftime('%m-%d-%Y'):(t - timedelta(days=1)).strftime('%m-%d-%Y')]
        X_train = new_pred_train.values
        truth.append(y_truth[t.strftime('%m-%d-%Y')])
        d_tree_predictions.append(y_pred[0])
        y_train = np.append(y_train, y_pred[0])
        dates.append(t.strftime('%m-%d-%Y'))

    # evaluate forecasts
    OLS_rmse = sqrt(mean_squared_error(truth, d_tree_predictions))
    predictions_df = prediction_aggregator(predictions_df, dates,
                                           d_tree_predictions, truth, feature, model_name)
    return predictions_df, OLS_rmse, model_name


def arima(state_covid_clean, train_start_date, train_end_date, test_prediction_start_date,
          test_range, feature, predictions_df, period):
    predict_train_df_date_range = state_covid_clean[train_start_date.strftime('%m-%d-%Y'):
                                                    (train_end_date - timedelta(days=1)).strftime('%m-%d-%Y')]
    predict_test_df_date_range = state_covid_clean[test_prediction_start_date.strftime('%m-%d-%Y'):
                                                   (test_prediction_start_date +
                                                    timedelta(days=test_range)).strftime('%m-%d-%Y')]

    train_df = predict_train_df_date_range[feature].shift(-period)
    train = train_df.values

    truth = []
    ARIMA_predictions = []
    dates = []

    # walk-forward validation (prediction)
    for t in pd.date_range(test_prediction_start_date, test_prediction_start_date + timedelta(days=test_range)):
        model = ARIMA(train, order=(5, 0, 5))
        model_fit = model.fit(method_kwargs={"warn_convergence": False})
        output = model_fit.forecast()
        yhat = output[0]
        train = np.append(train, yhat)
        true_value = predict_test_df_date_range.loc[t.strftime('%m-%d-%Y')][feature]
        truth.append(true_value)
        ARIMA_predictions.append(yhat)
        dates.append(t.strftime('%m-%d-%Y'))

    ARIMA_rmse = sqrt(mean_squared_error(truth, ARIMA_predictions))

    model_name = 'ARIMA'
    predictions_df = prediction_aggregator(predictions_df, dates,
                                           ARIMA_predictions, truth, feature, model_name)
    return predictions_df, ARIMA_rmse, model_name


def prediction_aggregator(predictions_df, dates, predictions, truth, feature, model_name):
    if predictions_df.empty:
        predictions_df = pd.DataFrame(data=zip(truth, predictions), index=dates,
                                      columns=[feature + '_truth', feature + '_' + model_name + '_predications'])
    else:
        # check for truth column
        truth_cols = [col for col in predictions_df.columns if feature + '_truth' in col]
        if len(truth_cols) == 1:  # truth already found
            temp_df = pd.DataFrame(data=zip(predictions), index=dates,
                                   columns=[feature + '_' + model_name + '_predications'])
        else:
            temp_df = pd.DataFrame(data=zip(truth, predictions), index=dates,
                                   columns=[feature + '_truth', feature + '_' + model_name + '_predications'])
        predictions_df = pd.concat([predictions_df, temp_df], axis=1)

    return predictions_df


def ensemble_df(states, sources, time_periods, beta_gamma):
    for state in states:
        # read in predictions
        predictions_df = pd.read_csv('.\\predictions\\%s_model_predictions.csv' % state, index_col='date')

        # read in stored rmse values for each model
        reader = csv.reader(open('.\\predictions\\%s_model_predictions_rmse.csv' % state, 'r'))
        rmse_aggregator = {}
        for row in reader:
            rmse_aggregator[row[0]] = float(row[1])

        models_list = list(rmse_aggregator.keys())
        for source in sources:
            for period in time_periods:
                for bg in beta_gamma:
                    ensemble_aggregate = predictions_df.copy(deep=True)
                    feature = state + '-' + source + '_' + bg + '_' + str(period)
                    model_cols = [model_name for model_name in models_list if feature in model_name]
                    ensemble_denominator = 0
                    for mod in model_cols:
                        ensemble_aggregate[mod + '_predications'] = \
                            (1 / rmse_aggregator[mod]) * ensemble_aggregate[mod + '_predications']
                        ensemble_denominator += 1 / rmse_aggregator[mod]
                    predictions_df[feature + '_ensemble'] = ensemble_aggregate.sum(axis=1) / ensemble_denominator

                    truth = predictions_df[feature + '_truth'].values
                    ensemble_predictions = predictions_df[feature + '_ensemble'].values
                    ensemble_rmse = sqrt(mean_squared_error(truth, ensemble_predictions))
                    rmse_aggregator[feature + '_' + 'ensemble'] = ensemble_rmse

        predictions_df.to_csv('.\\predictions\\%s_model_predictions.csv' % state)

        with open('.\\predictions\\%s_model_predictions_rmse.csv' % state, 'w') as f:
            for key in rmse_aggregator.keys():
                f.write("%s,%s\n" % (key, rmse_aggregator[key]))

    return predictions_df


def run_models(covid_national_df, states, sources, time_periods, beta_gamma, train_start_date, train_end_date,
               test_prediction_start_date):

    for state in states:
        predictions_df = pd.DataFrame()
        rmse_aggregator = {}
        for source in sources:
            for period in time_periods:
                for bg in beta_gamma:
                    feature = state + '-' + source + '_' + bg + '_' + str(period)
                    # ensure virgina does not include west virgina
                    if state == 'Virginia':
                        state_covid_cols = [col for col in covid_national_df.columns if state in col]
                        state_covid_cols = [state for state in state_covid_cols if 'West' not in state]
                    else:
                        state_covid_cols = [col for col in covid_national_df.columns if state in col]
                    state_covid_clean = covid_national_df[state_covid_cols]

                    # begin predictions
                    # Arima
                    predictions_df, rmse, model_name = arima(state_covid_clean, train_start_date, train_end_date,
                                                             test_prediction_start_date, predict_t_max, feature,
                                                             predictions_df, period)
                    rmse_aggregator[feature + '_' + model_name] = rmse

                    # OLS
                    predictions_df, rmse, model_name = ols(state_covid_clean, train_start_date, train_end_date,
                                                           test_prediction_start_date, predict_t_max, feature,
                                                           predictions_df, period)
                    rmse_aggregator[feature + '_' + model_name] = rmse

                    # tree regressors
                    # vanilla decision tree
                    predictions_df, rmse, model_name = d_tree(state_covid_clean, train_start_date, train_end_date,
                                                              test_prediction_start_date, predict_t_max, feature,
                                                              predictions_df, period,
                                                              DT=True, RT=False, Bagged_DT=False, Bagged_RT=False,
                                                              Boostrap=False,
                                                              n_estimators=10, max_depth=20)
                    rmse_aggregator[feature + '_' + model_name] = rmse

                    # random forest
                    predictions_df, rmse, model_name = d_tree(state_covid_clean, train_start_date, train_end_date,
                                                              test_prediction_start_date, predict_t_max, feature,
                                                              predictions_df, period,
                                                              DT=False, RT=True, Bagged_DT=False, Bagged_RT=False,
                                                              Boostrap=False,
                                                              n_estimators=10, max_depth=20)
                    rmse_aggregator[feature + '_' + model_name] = rmse

                    # bagged vanilla decision tree
                    predictions_df, rmse, model_name = d_tree(state_covid_clean, train_start_date, train_end_date,
                                                              test_prediction_start_date, predict_t_max, feature,
                                                              predictions_df, period,
                                                              DT=False, RT=False, Bagged_DT=True, Bagged_RT=False,
                                                              Boostrap=True,
                                                              n_estimators=10, max_depth=20)
                    rmse_aggregator[feature + '_' + model_name] = rmse

                    # bagged random forest
                    predictions_df, rmse, model_name = d_tree(state_covid_clean, train_start_date, train_end_date,
                                                              test_prediction_start_date, predict_t_max, feature,
                                                              predictions_df, period,
                                                              DT=False, RT=False, Bagged_DT=False, Bagged_RT=True,
                                                              Boostrap=True,
                                                              n_estimators=10, max_depth=20)
                    rmse_aggregator[feature + '_' + model_name] = rmse

        predictions_df.index.name = 'date'
        predictions_df.to_csv('.\\predictions\\%s_model_predictions.csv' % state)

        with open('.\\predictions\\%s_model_predictions_rmse.csv' % state, 'w') as f:
            for key in rmse_aggregator.keys():
                f.write("%s,%s\n" % (key, rmse_aggregator[key]))


def b_g_stats(states, sources, time_periods, beta_gamma):

    composite_bg = []
    models = []

    for state in states:
        # read in stored rmse values for each model
        reader = csv.reader(open('.\\predictions\\%s_model_predictions_rmse.csv' % state, 'r'))
        rmse_aggregator = {}
        for row in reader:
            rmse_aggregator[row[0]] = float(row[1])
        models_list = list(rmse_aggregator.keys())
        models = [model_name.replace('%s-' % state, '') for model_name in models_list]
        rmse_values = list(rmse_aggregator.values())
        composite_bg.append(rmse_values)

    prediction_stats_states = pd.DataFrame(data=composite_bg, index=states, columns=models)
    prediction_stats_states['State Mean'] = prediction_stats_states.mean(axis=1)
    prediction_stats_states.loc['Model Mean'] = prediction_stats_states.mean()
    prediction_stats_states.index.name = 'State'
    prediction_stats_states.to_csv('.\\predictions\\aggregate_prediction_stats_predictions.csv')

    ensemble_time_cols = []
    for time in time_periods:
        ensemble_cols = [col for col in prediction_stats_states.columns if 'ensemble' in col]
        ensemble_cols = [col for col in ensemble_cols if str(time) in col]
        for col in ensemble_cols:
            ensemble_time_cols.append(col)

    prediction_stats_states[ensemble_time_cols].loc['Model Mean'].to_csv(
        '.\\predictions\\aggregate_model_stats_predictions.csv')

    for bg in beta_gamma:
        b_g_cols = [col for col in ensemble_time_cols if bg in col]
        prediction_stats_states[b_g_cols].loc['Model Mean'].to_csv(
            '.\\predictions\\aggregate_model_%s_stats_predictions.csv' % bg)

    for time in time_periods:
        for source in sources:
            time_models = [col for col in prediction_stats_states.columns if str(time) in col]
            time_source_models = [col for col in time_models if source in col]
            prediction_stats_states[time_source_models].to_csv(
                '.\\predictions\\aggregate_model_time-%s_source-%s_stats_predictions.csv'
                % (time, source))


if __name__ == "__main__":

    # load covid national dataset
    covid_national_df = pd.read_csv(".\\Data\\US_data_combined_clean.csv", index_col='date')

    predict_t_max = 30

    # split for train and test dates
    train_start_date = datetime.date(2021, 1, 1)
    train_end_date = datetime.date(2021, 7, 1)
    test_prediction_start_date = train_end_date + timedelta(days=1)

    states = get_state_list()
    sources = ['jh', 'cdc']
    time_periods = [7, 14, 30]
    beta_gamma = ['beta', 'gamma']

    # generate prediction data
    run_models(covid_national_df, states, sources, time_periods, beta_gamma, train_start_date, train_end_date,
               test_prediction_start_date)

    # generate ensemble data
    ensemble_df(states, sources, time_periods, beta_gamma)

    # generate beta / gamma stats for report
    b_g_stats(states, sources, time_periods, beta_gamma)
