# Term Project for CSE-8803 EPI (Data Science for Epidemiology) #
## Team Members ##
| Name           | GTID      |
|----------------|-----------|
| Whitaker Chu   | `mchu31`  |
| Kai Ouyang     | `ko40`    |
| Nicholas Saney | `nsaney3` |

# Web Page #
- https://github.gatech.edu/pages/nsaney3/2021-fall-cse8803epi-project/

# How to Run #
## Data cleaning ##
Enter the `src/Data` folder.

1) run python US_states_mobility.py This generates combines the mobility data from data and google
2) run python US_covid_data.py This creates the initial dataset of covid infections and deaths per state. This also normalizes the infections and deaths ratio based on population.
3) run python US_combine_data.py This combines the US mobility data and infections and deaths per state in a time series.
4) run US beta_gamma_calculator.py This calculates beta and gamma in 7, 14, 30 day lookback windows for each day from Jan 1 to August 31.

## Forecasting ##
Enter the `src` folder.

1) run `bg_forecast.py` to forecast for a time period. Note, this process takes a long time (>60 minutes). You can test which parts of the function you want to test by commenting out sections of the code:     
    a. generate prediction data
    ```
    run_models(covid_national_df, states, sources, time_periods, beta_gamma, train_start_date, train_end_date,
    test_prediction_start_date)
    ```

    b. generate ensemble data
    ```
    ensemble_df(states, sources, time_periods, beta_gamma)
    ```

    c. generate beta / gamma stats for report
    ```
    b_g_stats(states, sources, time_periods, beta_gamma)
    ```

    Note: prediction date + t_max cannot exceed August 31, 2021 as that is the end of the data we cleaned.

2) Run `TVSIR_forecasting.py` to generate TV-ODE death and infection rate predictions using the predicted Gamma and Beta Values
3) Run `ODESIR_forecasting.py` to generate standard ODE predictions for infection and death rates.
4) Run `TV_vs_SIR.py to` compare the results generated from steps 2 and 3 accross the nation. This will generate 4 images for each state in the "Images" directory.

