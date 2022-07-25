from numpy.lib.shape_base import get_array_wrap
import pandas as pd
import numpy as np
import datetime

start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2021, 8, 31)
dates_of_interest = pd.date_range(start=start, end=end)

apple_df = pd.read_csv(".\\Mobility_Data\\Apple_Data\\AppleMobilityTrends.csv")
# county_df = pd.read_excel(".\\Population_Data\\PopulationByCounty.xlsx", names=['Location', 'Population'])[
#             1:]  # Ditch the first row, as it is just header info
google_df = pd.read_csv(".\\Mobility_Data\\Google_Data\\2021_US_Region_Mobility_Report.csv")

# clean list of US states. No US territories.
states_us_list = google_df['sub_region_1'].unique()
states_us_list = [state for state in states_us_list if pd.isnull(state) == False]  # remove null
states_us_list = [state for state in states_us_list if state != 'District of Columbia']  # remove Washington DC

# clean Apple transit data.
# our "clean" data is not all that clean. It has PR and VI as states.
# Also, we are missing some transit data from 6 states.
apple_us_df = apple_df[apple_df['country'] == 'United States']  # select country as USA
apple_us_df = apple_us_df[apple_us_df['geo_type'] == 'sub-region']  # select states only
apple_us_df = apple_us_df[apple_us_df['region'].isin(states_us_list)]  # no us territories
apple_us_df = apple_us_df.drop(columns=['geo_type', 'alternative_name', 'sub-region', 'country'])  # drop unused columns

# clean Google transit data.
# select states only - no counties
google_us_df = google_df[google_df['sub_region_2'].isna()]
google_us_df = google_us_df[google_us_df['sub_region_1'].isin(states_us_list)]

# select google columns of interest:
google_us_df = google_us_df[['sub_region_1', 'date',
                             'retail_and_recreation_percent_change_from_baseline',
                             'grocery_and_pharmacy_percent_change_from_baseline',
                             'parks_percent_change_from_baseline',
                             'transit_stations_percent_change_from_baseline',
                             'workplaces_percent_change_from_baseline',
                             'residential_percent_change_from_baseline']]

# clean up column names
apple_us_df['region'] = apple_us_df['region'].astype(str) + '-apple'
apple_us_df['state_mobility'] = apple_us_df[['region', 'transportation_type']].agg('-'.join, axis=1)

google_us_df.rename(columns={'sub_region_1': 'State', 'date': 'Date',
                             'retail_and_recreation_percent_change_from_baseline': 'retail_and_recreation',
                             'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_and_pharmacy',
                             'parks_percent_change_from_baseline': 'parks',
                             'transit_stations_percent_change_from_baseline': 'transit_stations',
                             'workplaces_percent_change_from_baseline': 'workplaces',
                             'residential_percent_change_from_baseline': 'residential'
                             }, inplace=True)

# drop unwanted google data:
google_dates_of_interest_formatted = (dates_of_interest.strftime('%Y-%m-%d')).tolist()
google_clean_init = True
google_us_clean_df = pd.DataFrame

for state in states_us_list:
    google_state_df = google_us_df[google_us_df['State'] == state]
    google_state_df = google_state_df[google_state_df['Date'].isin(google_dates_of_interest_formatted)]
    google_state_df = google_state_df.drop(columns=['State'])  # drop unused columns
    google_state_df = google_state_df.set_index('Date')
    google_state_df = google_state_df.add_prefix(state + '-google-')
    if google_clean_init:
        google_us_clean_df = google_state_df
        google_clean_init = False
    else:
        google_us_clean_df = pd.concat([google_us_clean_df, google_state_df], axis=1)

# drop unwanted apple data:
apple_dates_of_interest_formatted = (dates_of_interest.strftime('%Y-%m-%d')).tolist()
apple_dates_of_interest_formatted.insert(0, 'state_mobility')
apple_us_df = apple_us_df[apple_us_df.columns.intersection(apple_dates_of_interest_formatted)]

# get clean apple data
clean_index = apple_us_df['state_mobility'].values.tolist()
apple_us_clean_df = apple_us_df.drop(columns=['state_mobility'])
apple_us_clean_df = apple_us_clean_df.T
apple_us_clean_df.set_axis(clean_index, axis=1, inplace=True)

# drop apple transit data because many states do not have it
apple_transit_col = [col for col in apple_us_clean_df.columns if 'transit' in col]
apple_us_clean_df = apple_us_clean_df.drop(columns=apple_transit_col)

# export to csv
apple_us_clean_df.to_csv('apple_us_clean.csv')
google_us_clean_df.to_csv('google_us_clean.csv')
app_and_g_us_clean_df = pd.concat([apple_us_clean_df, google_us_clean_df], axis=1)

# standardize format of index datetime
app_and_g_us_clean_df.index.name = 'date'
app_and_g_us_clean_df.index = pd.to_datetime(app_and_g_us_clean_df.index, format='%Y-%m-%d').strftime('%m-%d-%Y')

# ffill and bfill
app_and_g_us_clean_df = app_and_g_us_clean_df.ffill().bfill()

# out to csv
app_and_g_us_clean_df.to_csv('app_and_g_us_clean_df.csv')

