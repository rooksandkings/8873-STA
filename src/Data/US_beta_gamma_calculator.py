import numpy as np
import pandas as pd
from scipy.integrate import ode, solve_ivp
import scipy.optimize as optimize
import datetime

def sir_ode(init, parms):
    b, g = parms
    S, I, R = init

    # ODEs
    dS = -b * S * I
    dI = b * S * I - g * I
    dR = g * I
    return [dS, dI, dR]


def beta_gamma(data):

    # setup for scipy minimize)
    initial_guess = [.025, .00001]
    result = optimize.minimize(minimize_this_function, initial_guess, bounds=((0, 1), (0, 1)), args=data)

    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    parms = [fitted_params[0], fitted_params[1]]

    return parms


def minimize_this_function(params, data):

    I_0 = data.iloc[:, 0][0]
    R_0 = data.iloc[:, 1][0]
    S_0 = 1 - I_0 - R_0
    init = [S_0, I_0, R_0]

    if S_0 < 0:
        print("Initial values do not add to 1")

    SIR_ground_I_t = data.iloc[:, 0].values
    SIR_ground_R_t = data.iloc[:, 1].values
    SIR_ground_S_t = 1 - SIR_ground_R_t - SIR_ground_I_t

    t_max = data.shape[0]
    times = np.arange(t_max)

    parms = [params[0], params[1]]

    sir_sol = solve_ivp(fun=lambda t, y: sir_ode(y, parms), t_span=[min(times), max(times)], y0=init, t_eval=times)
    sir_out = pd.DataFrame({"t": sir_sol["t"], "S": sir_sol["y"][0], "I": sir_sol["y"][1], "R": sir_sol["y"][2]})

    return mean_squared_error(sir_out['R'].values, SIR_ground_R_t)


def mean_squared_error(x_0, x_1):

    return np.square(np.subtract(x_0, x_1)).mean()


if __name__ == "__main__":

    start_date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2021, 8, 31)

    covid_US_df = pd.read_csv("covid_US_data_clean.csv", index_col='date')

    # clean list of US states. No US territories.
    google_df = pd.read_csv(".\\Mobility_Data\\Google_Data\\2021_US_Region_Mobility_Report.csv")
    states_us_list = google_df['sub_region_1'].unique()
    states_us_list = [state for state in states_us_list if pd.isnull(state) == False]  # remove null
    states_us_list = [state for state in states_us_list if state != 'District of Columbia']  # remove Washington DC

    intial_df = True

    for state in states_us_list:

        selected_state = state

        # ensure virgina does not include west virgina
        if state == 'Virginia':
            state_covid_cols = [col for col in covid_US_df.columns if state in col]
            state_covid_cols = [state for state in state_covid_cols if 'West' not in state]
        else:
            state_covid_cols = [col for col in covid_US_df.columns if state in col]

        state_covid_df = covid_US_df[state_covid_cols]
        state_covid_cols_ratio_cols = [col for col in state_covid_df.columns if 'ratio' in col]
        state_covid_ratio_df = state_covid_df[state_covid_cols_ratio_cols]

        # get infection and death data for CDC and JH
        state_covid_cols_ratio_jh_cols = [col for col in state_covid_ratio_df.columns if 'jh' in col]
        state_covid_ratio_jh_df = state_covid_ratio_df[state_covid_cols_ratio_jh_cols]
        state_covid_cols_ratio_cdc_cols = [col for col in state_covid_ratio_df.columns if 'cdc' in col]
        state_covid_ratio_cdc_df = state_covid_ratio_df[state_covid_cols_ratio_cdc_cols]

        delta = datetime.timedelta(days=1)
        iter_start_date = start_date
        iter_end_date = end_date
        beta_gamma_intervals = [7, 14, 30]

        # shift data by one to ensure we are not using current day's data in forecasting
        state_covid_ratio_jh_shift_df = state_covid_ratio_jh_df.shift(1)
        state_covid_ratio_cdc_shift_df = state_covid_ratio_cdc_df.shift(1)

        state_bg_list = []
        column_list = ['date']
        while iter_start_date <= iter_end_date:
            bg_list = [iter_start_date.strftime("%m-%d-%Y")]
            for date_int in beta_gamma_intervals:
                start_search_date = (iter_start_date - datetime.timedelta(days=date_int)).strftime("%m-%d-%Y")
                end_search_date = iter_start_date.strftime("%m-%d-%Y")
                jh_search_interval = state_covid_ratio_jh_shift_df[start_search_date: end_search_date]
                cdc_search_interval = state_covid_ratio_cdc_shift_df[start_search_date: end_search_date]
                jh_beta, jh_gamma = beta_gamma(jh_search_interval)
                cdc_beta, cdc_gamma = beta_gamma(cdc_search_interval)
                bg_list.extend([jh_beta, jh_gamma, cdc_beta, cdc_gamma])

            state_bg_list.append(bg_list)
            iter_start_date += delta

        for date_int in beta_gamma_intervals:
            column_list.extend([selected_state + '-jh_beta_' + str(date_int),
                                selected_state + '-jh_gamma_' + str(date_int),
                                selected_state + '-cdc_beta_' + str(date_int),
                                selected_state + '-cdc_gamma_' + str(date_int)])

        if intial_df:
            bg_df = pd.DataFrame(state_bg_list, columns=column_list).set_index('date', drop=True)
            intial_df = False
        else:
            state_df = pd.DataFrame(state_bg_list, columns=column_list).set_index('date', drop=True)
            bg_df = pd.concat([bg_df, state_df], axis=1)

    bg_df.to_csv('beta_gamma_US_clean.csv')


