from numpy.lib.shape_base import get_array_wrap
import pandas as pd
import numpy as np
import datetime

# clean list of US states. No US territories.
google_df = pd.read_csv(".\\Mobility_Data\\Google_Data\\2021_US_Region_Mobility_Report.csv")

states_us_list = google_df['sub_region_1'].unique()
states_us_list = [state for state in states_us_list if pd.isnull(state) == False]  # remove null
states_us_list = [state for state in states_us_list if state != 'District of Columbia']  # remove Washington DC

# combine all data

# read in all data
mobility_df = pd.read_csv("app_and_g_us_clean_df.csv", index_col='date')
bg_df = pd.read_csv("beta_gamma_US_clean.csv", index_col='date')
covid_df = pd.read_csv("covid_US_data_clean.csv", index_col='date')

start_date = (datetime.date(2021, 1, 1)).strftime('%m-%d-%Y')
end_date = (datetime.date(2021, 8, 31)).strftime('%m-%d-%Y')

# only take the dates we want from covid data
covid_df = covid_df[start_date: end_date]

# merge dataframes
us_all_data_df = pd.concat([mobility_df, bg_df], axis=1)
us_all_data_df = pd.concat([us_all_data_df, covid_df], axis=1)

# check data integrity
state_covid_cols_count = [col for col in us_all_data_df.columns if 'Georgia' in col]
for state in states_us_list:
    selected_state = state
    # ensure virgina does not include west virgina
    if state == 'Virginia':
        state_covid_cols = [col for col in us_all_data_df.columns if state in col]
        state_covid_cols = [state for state in state_covid_cols if 'West' not in state]
    else:
        state_covid_cols = [col for col in us_all_data_df.columns if state in col]

    if len(state_covid_cols) != len(state_covid_cols_count):
        print('DATA ERROR: State column features not equal')

# output data
us_all_data_df.to_csv('US_data_combined_clean.csv')

# output individual state data
for state in states_us_list:
    selected_state = state
    # ensure virgina does not include west virgina
    if state == 'Virginia':
        state_covid_cols = [col for col in us_all_data_df.columns if state in col]
        state_covid_cols = [state for state in state_covid_cols if 'West' not in state]
    else:
        state_covid_cols = [col for col in us_all_data_df.columns if state in col]

    us_all_data_df[state_covid_cols].to_csv('./State_Data_Clean/%s_combined_clean.csv' % state)

