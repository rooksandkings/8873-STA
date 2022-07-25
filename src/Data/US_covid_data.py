from os import listdir
from os.path import isfile, join
import pandas as pd
import csv
import datetime

def is_date_of_interest(date):
    start = datetime.datetime(2020, 11, 1).date()
    end = datetime.datetime(2021, 9, 30).date()
    return start <= date <= end
# this dictionary will have elements of the following format:
# {
#    '2021-01-01' : {
#       'jhu': {
#           'confirmed': { 'Georgia': 100, 'Alabama' : 100 ...}
#           'death': { 'Georgia': 2, 'Alabama' : 2 ...}
#       }
#       'cdc':{
#           'confirmed': { 'Georgia': 100, 'Alabama' : 100 ...}
#           'death': { 'Georgia': 2, 'Alabama' : 2 ...}
#       },
#       'population': { 'Georgia': 1000, 'Alabama': 1000 ... }
#   }
# }
covid_data_dictionary = {}


# https://code.activestate.com/recipes/577305-python-dictionary-of-us-states-and-territories/
state_code_dictionary = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        # 'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        # 'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        # 'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        # 'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        # 'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        # 'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        # 'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


jh_covid_dDirectory_path = 'JH_Covid_Data'

# get all JHU Covid data file names, and therefore their dates
# there's a random README file in this folder LOL
jh_covid_file_names = [f for f in listdir(jh_covid_dDirectory_path) if isfile(join(jh_covid_dDirectory_path, f)) and f != 'README.md']

# we only care about US states
regions_to_exclude = ['American Samoa', 
                    'Diamond Princess', 
                    'District of Columbia', 
                    'Grand Princess',
                    'Guam',
                    'Northern Mariana Islands',
                    'Puerto Rico',
                    'Virgin Islands',
                    ]

# populate the result dictionary with JHU data
for jh_covid_file_name in jh_covid_file_names:
    current_date_string = jh_covid_file_name.strip('.csv')
    current_date_obj = datetime.datetime.strptime(current_date_string, '%m-%d-%Y').date()    
    if is_date_of_interest(current_date_obj):
        current_jhu_file = open("JH_Covid_Data/"+jh_covid_file_name)
        current_jhu_csv_reader = csv.reader(current_jhu_file)
        header = next(current_jhu_csv_reader) # just to move the iterator past the header
        for row in current_jhu_csv_reader:
            current_state_full_name = row[0]
            if current_state_full_name not in regions_to_exclude:
                if(current_date_obj in covid_data_dictionary):
                    current_confirmed_dictionary = covid_data_dictionary[current_date_obj]['jhu']['confirmed']
                    current_confirmed_dictionary[current_state_full_name] = int(row[5]) # header "Confirmed"
                    current_death_dictionary = covid_data_dictionary[current_date_obj]['jhu']['death']
                    current_death_dictionary[current_state_full_name] = int(row[6]) # header "Deaths"
                else:
                    new_dict = {
                        'confirmed' : { current_state_full_name: int(row[5]) }, # header "Confirmed"
                        'death': {current_state_full_name : int(row[6]) }, # header "Deaths"
                    }
                    covid_data_dictionary[current_date_obj] = { 'jhu': new_dict}
        current_jhu_file.close()

# populate the result dictionary with CDC data
cdc_covid_data_file = open("CDC_Covid_Data/covid_data.csv")
cdc_covid_data_csv_reader = csv.reader(cdc_covid_data_file)
header = next(cdc_covid_data_csv_reader) # again, just getting past the header of the csv
for row in cdc_covid_data_csv_reader:
    current_date_string = row[0].replace('/', '-') # convert from "02/04/2021" to "02-04-2021"
    current_date_obj = datetime.datetime.strptime(current_date_string, '%m-%d-%Y').date()    

    if is_date_of_interest(current_date_obj):
        current_state_full_name = row[1]
        if current_state_full_name in state_code_dictionary:
            current_state_full_name = state_code_dictionary[current_state_full_name]
            if 'cdc' in covid_data_dictionary[current_date_obj]:
                existing_confirmed_entry = covid_data_dictionary[current_date_obj]['cdc']['confirmed']
                existing_confirmed_entry[current_state_full_name] = int(row[2]) # header "tot_cases"
                existing_death_entry = covid_data_dictionary[current_date_obj]['cdc']['death']
                existing_death_entry[current_state_full_name] = int(row[7]) # header "tot_death"
            else:
                new_dict = {
                        'confirmed' : { current_state_full_name: int(row[2]) }, # header "tot_cases"
                        'death': {current_state_full_name : int(row[7]) }, # header "tot_death"
                }
                existing_date_dictionary = covid_data_dictionary[current_date_obj] 
                # the jhu part is already in existing_date_dictionary, so insert new key instead of initializing the whole dictionary
                existing_date_dictionary['cdc'] = new_dict
cdc_covid_data_file.close()

# populate the population data from census
population_df = pd.read_excel("Population_Data/PopulationByState_Census.xlsx",
                            header = 8, # Ditch the first few rows we don't care about
                            skipfooter = 8, # Ditch the last few footer rows we don't care about 
                            usecols = "A,M",  # we only care about column A (state) and M (population in 2019)
                            names=['Location', 'Population'])
population_dictionary = {}
for ind in population_df.index:
    location = population_df['Location'][ind].strip('.') # the state names have a leading dot we need to remove
    if location not in regions_to_exclude:
        population_dictionary[location] = population_df['Population'][ind]
# the excel already lists the states in alphabetical order so we don't need to sort population_dictionary any more
# this is also the order of states we will respect


# generate CSV

# header_list = ['date']
# for state_full_name in population_dictionary.keys():
#         header_list.append('census-' + state_full_name + '-pop')
# for state_full_name in population_dictionary.keys():
#         header_list.append('jh-' + state_full_name + '-confirmed')
# for state_full_name in population_dictionary.keys():
#         header_list.append('jh-' + state_full_name + '-death')
# for state_full_name in population_dictionary.keys():
#         header_list.append('cdc-' + state_full_name + '-confirmed')
# for state_full_name in population_dictionary.keys():
#         header_list.append('cdc-' + state_full_name + '-death')
# for state_full_name in population_dictionary.keys():
#         header_list.append('jh-' + state_full_name + '-confirmed_ratio')
# for state_full_name in population_dictionary.keys():
#         header_list.append('jh-' + state_full_name + '-death_ratio')
# for state_full_name in population_dictionary.keys():
#         header_list.append('cdc-' + state_full_name + '-confirmed_ratio')
# for state_full_name in population_dictionary.keys():
#         header_list.append('cdc-' + state_full_name + '-death_ratio')

header_list = ['date']
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-census_pop')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-jh_confirmed')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-jh_death')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-cdc_confirmed')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-cdc_death')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-jh_confirmed_ratio')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-jh_death_ratio')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-cdc_confirmed_ratio')
for state_full_name in population_dictionary.keys():
        header_list.append(state_full_name + '-cdc_death_ratio')

final_list = [header_list]

# go through the dates in order to ensure the row order in the csv
start_date = datetime.date(2020, 11, 1)
end_date = datetime.date(2021, 9, 30)
delta = datetime.timedelta(days=1)
while start_date <= end_date:
    current_list = [start_date.strftime("%m-%d-%Y")]

    # populate population data by states
    for state_full_name in population_dictionary.keys():
        state_population = population_dictionary[state_full_name]
        current_list.append(state_population)

    # populate jh confirmed cases by states
    for state_full_name in population_dictionary.keys():
        jhu_confirmed = covid_data_dictionary[start_date]['jhu']['confirmed'][state_full_name]
        current_list.append(jhu_confirmed)

    # populate jh death cases by states
    for state_full_name in population_dictionary.keys():
        jhu_death = covid_data_dictionary[start_date]['jhu']['death'][state_full_name]
        current_list.append(jhu_death)
    
    # populate cdc confirmed cases by states
    for state_full_name in population_dictionary.keys():
        cdc_confirmed = covid_data_dictionary[start_date]['cdc']['confirmed'][state_full_name]
        current_list.append(cdc_confirmed)

    # populate cdc death cases by states
    for state_full_name in population_dictionary.keys():
        cdc_death = covid_data_dictionary[start_date]['cdc']['death'][state_full_name]
        current_list.append(cdc_death)
    
    # populate jh confirmed % by states
    for state_full_name in population_dictionary.keys():
        jhu_confirmed = covid_data_dictionary[start_date]['jhu']['confirmed'][state_full_name]
        state_population = population_dictionary[state_full_name]
        jhu_confirmed_percentage = jhu_confirmed / state_population
        current_list.append(jhu_confirmed_percentage)
    
    # populate jh death % by states
    for state_full_name in population_dictionary.keys():
        jhu_death = covid_data_dictionary[start_date]['jhu']['death'][state_full_name]
        state_population = population_dictionary[state_full_name]
        jhu_death_percentage = jhu_death / state_population
        current_list.append(jhu_death_percentage)

    # populate cdc confirmed % by states
    for state_full_name in population_dictionary.keys():
        cdc_confirmed = covid_data_dictionary[start_date]['cdc']['confirmed'][state_full_name]
        state_population = population_dictionary[state_full_name]
        cdc_confirmed_percentage = cdc_confirmed / state_population
        current_list.append(cdc_confirmed_percentage)

    # populate cdc death % by states
    for state_full_name in population_dictionary.keys():
        cdc_death = covid_data_dictionary[start_date]['cdc']['death'][state_full_name]
        state_population = population_dictionary[state_full_name]
        cdc_death_percentage = cdc_death / state_population
        current_list.append(cdc_death_percentage)

    final_list.append(current_list)
    start_date += delta

# write to the csv file
with open("covid_US_data_clean.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(final_list)

