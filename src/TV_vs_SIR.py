from os import stat
import os.path
from datetime import timedelta
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt



def main():
    state_data_glob_path = os.path.join('.', 'Data', 'State_Data_Clean', '*')
    for cleaned_data_path in sorted(glob(state_data_glob_path)):
        state_name = os.path.basename(cleaned_data_path).split('_', 1)[0]
        
        predicted_TV_data_path = os.path.join('.', 'predictions', state_name + "_tvsir_predictions.csv")
        predicted_SIR_data_path = os.path.join('.', 'predictions', state_name + "_ODEsir_predictions.csv")

        plot_deaths14(predicted_SIR_data_path, predicted_TV_data_path, state_name)
        plot_deaths7(predicted_SIR_data_path, predicted_TV_data_path, state_name)
        plot_infections14(predicted_SIR_data_path, predicted_TV_data_path, state_name)
        plot_infections7(predicted_SIR_data_path, predicted_TV_data_path, state_name)

def plot_deaths7(predicted_SIR_data_path, predicted_TV_data_path, state_name):
    df_SIR = pd.read_csv(predicted_SIR_data_path, index_col='date', parse_dates=['date'])
    df_TV = pd.read_csv(predicted_TV_data_path, index_col='date', parse_dates=['date'])
    
    deaths_SIR = df_SIR["ODE_Death_Prediction"]
    deaths_TV = df_TV[state_name + "-cdc_bg7_mortality_ratio_predicted"]
    deaths_actual = df_TV[state_name + "-cdc_death_ratio"]

    x = range(0,7)
    plt.xlabel('Days Predicted')
    plt.ylabel('R(t) (death proportion)')
    plt.title('7 Day Death Predictions for ' + state_name)
    plt.plot(x, deaths_SIR[:7], color='blue', label="SIR Predicted Deaths")
    plt.plot(x, deaths_TV[:7], color='red', label="TV-SIR Predicted Deaths")
    plt.plot(x, deaths_actual[:7], color='green', label="Actual Deaths")
    plt.legend(loc="upper left")
    plt.savefig("Images\\" + state_name + '_deaths_7.png')
    plt.show()

def plot_deaths14(predicted_SIR_data_path, predicted_TV_data_path, state_name):
    df_SIR = pd.read_csv(predicted_SIR_data_path, index_col='date', parse_dates=['date'])
    df_TV = pd.read_csv(predicted_TV_data_path, index_col='date', parse_dates=['date'])
    
    deaths_SIR = df_SIR["ODE_Death_Prediction"]
    deaths_TV = df_TV[state_name + "-cdc_bg14_mortality_ratio_predicted"]
    deaths_actual = df_TV[state_name + "-cdc_death_ratio"]

    x = range(0,14)
    plt.xlabel('Days Predicted')
    plt.ylabel('R(t) (death proportion)')
    plt.title('14 Day Death Predictions for ' + state_name)
    plt.plot(x, deaths_SIR[:14], color='blue', label="SIR Predicted Deaths")
    plt.plot(x, deaths_TV[:14], color='red', label="TV-SIR Predicted Deaths")
    plt.plot(x, deaths_actual[:14], color='green', label="Actual Deaths")
    plt.legend(loc="upper left")
    plt.savefig("Images\\" + state_name + '_deaths_14.png')
    plt.show()

def plot_infections7(predicted_SIR_data_path, predicted_TV_data_path, state_name):
    df_SIR = pd.read_csv(predicted_SIR_data_path, index_col='date', parse_dates=['date'])
    df_TV = pd.read_csv(predicted_TV_data_path, index_col='date', parse_dates=['date'])
    
    inf_SIR = df_SIR["ODE_Infection_Prediction"]
    inf_TV = df_TV[state_name + "-cdc_bg7_cases_ratio_predicted"]
    inf_actual = df_TV[state_name + "-cdc_confirmed_ratio"]

    x = range(0,7)
    plt.xlabel('Days Predicted')
    plt.ylabel('I(t) (infection proportion)')
    plt.title('7 Day Infection Predictions for ' + state_name)
    plt.plot(x, inf_SIR[:7], color='blue', label="SIR Predicted Infections")
    plt.plot(x, inf_TV[:7], color='red', label="TV-SIR Predicted Infections")
    plt.plot(x, inf_actual[:7], color='green', label="Actual Infections")
    plt.legend(loc="upper left")
    plt.savefig("Images\\" + state_name + '_infections_7.png')
    plt.show()

def plot_infections14(predicted_SIR_data_path, predicted_TV_data_path, state_name):
    df_SIR = pd.read_csv(predicted_SIR_data_path, index_col='date', parse_dates=['date'])
    df_TV = pd.read_csv(predicted_TV_data_path, index_col='date', parse_dates=['date'])
    
    inf_SIR = df_SIR["ODE_Infection_Prediction"]
    inf_TV = df_TV[state_name + "-cdc_bg14_cases_ratio_predicted"]
    inf_actual = df_TV[state_name + "-cdc_confirmed_ratio"]

    x = range(0,14)
    plt.xlabel('Days Predicted')
    plt.ylabel('I(t) (infection proportion)')
    plt.title('14 Day Infection Predictions for ' + state_name)
    plt.plot(x, inf_SIR[:14], color='blue', label="SIR Predicted Infections")
    plt.plot(x, inf_TV[:14], color='red', label="TV-SIR Predicted Infections")
    plt.plot(x, inf_actual[:14], color='green', label="Actual Infections")
    plt.legend(loc="upper left")
    plt.savefig("Images\\" + state_name + '_infections_14.png')
    plt.show()
    



if __name__ == '__main__':
    main()
