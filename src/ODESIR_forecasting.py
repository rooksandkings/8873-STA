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
        print(state_name)
        predict_ODE_SIR(cleaned_data_path, predicted_data_path, state_name)
        
# Calculate delta at each time step
def calculate_dS(beta_val,  S, I):
    return -1.0 * (beta_val * S * I)

def calculate_dI(beta_val,gamma_val, S, I):
    return (1.0 * (beta_val * S * I)) - (gamma_val * I)

def calculate_dR(gamma_val, I):
    return 1.0 * (gamma_val * I)

def predict_ODE_SIR(cleaned_data_path, predicted_data_path, state_name):
    output_data_path = predicted_data_path.replace('_model_predictions.csv', '_ODEsir_predictions.csv')
    output_rmse_path = output_data_path.replace('.csv', '_rmse.csv')
    #load data
    df_cleaned = pd.read_csv(cleaned_data_path, index_col='date', parse_dates=['date'])

    i = df_cleaned[state_name + "-cdc_confirmed_ratio"]
    r = df_cleaned[state_name + "-cdc_death_ratio"]

    #start predicting for july 2 2021
    i0 = i.loc['2021-07-02']
    r0 = r.loc['2021-07-02']
    s0 = 1.0 - (i0 + r0)
    
    # 
    beta = df_cleaned[state_name + "-cdc_beta_7"].loc['2021-07-02']
    gamma = df_cleaned[state_name + "-cdc_gamma_7"].loc['2021-07-02']

    s = s0
    i = i0
    r = r0

    column_names = ["ODE_Death_Prediction", "ODE_Infection_Prediction", "date"]
    ODE_df = pd.DataFrame(columns = column_names)

    day = 2
    for idx in range(0, 14):
        # Compute Delta for the iteration
        dS = calculate_dS(beta_val=beta, S=s, I=i)
        dI = calculate_dI(beta_val=beta, gamma_val=gamma, S=s, I=i)
        dR = calculate_dR(gamma_val=gamma, I=i)

        # Add the delta on to get the legitimate SIR values
        s = s + dS
        i = i + dI
        r = r + dR
        date = "2021-07-0" + str(day + idx) if idx < 8 else "2021-07-" + str(day + idx)
        data_dict = {"ODE_Death_Prediction" : r + dR, "ODE_Infection_Prediction" : i + dI, "date" : date}

        ODE_df = ODE_df.append(data_dict, ignore_index = True)
    
    ODE_df['date'] = pd.to_datetime(ODE_df['date'], format="%Y-%m-%d")
    ODE_df = ODE_df.set_index("date")
    
    print('saving predictions to: ', output_data_path)
    ODE_df.to_csv(output_data_path)
    



if __name__ == '__main__':
    main()