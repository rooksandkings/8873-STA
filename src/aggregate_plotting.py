from os import stat
import os.path
from datetime import timedelta
from glob import glob
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt



def main():
    beta_path = os.path.join('.', 'predictions',"aggregate_model_beta_stats_predictions.csv")
    gamma_path = os.path.join('.', 'predictions',"aggregate_model_gamma_stats_predictions.csv")

    df_beta = pd.read_csv(beta_path)
    df_gamma = pd.read_csv(gamma_path)

    plt.style.use('ggplot')
    x = ["JH 7", "CDC 7", "JH 14", "CDC 14", "JH 30", "CDC 30"]
    mean = df_beta["Model Mean"]
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, mean, color='green')
    plt.xlabel("Model")
    plt.ylabel("Mean Beta")
    plt.title("Aggregated Beta Values for Ensemble Models")
    plt.xticks(x_pos, x)
    plt.xticks(fontsize=12)
    plt.savefig("Images\\Mean_Aggregated_Beta")
    plt.show()

    plt.style.use('ggplot')
    x = ["JH 7", "CDC 7", "JH 14", "CDC 14", "JH 30", "CDC 30"]
    mean = df_gamma["Model Mean"]
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, mean, color='green')
    plt.xlabel("Model")
    plt.ylabel("Mean Gamma")
    plt.title("Aggregated Gamma Values for Ensemble Models")
    plt.xticks(x_pos, x)
    plt.xticks(fontsize=12)
  

    plt.savefig("Images\\Mean_Aggregated_Gamma")
    plt.show()




if __name__ == '__main__':
    main()
