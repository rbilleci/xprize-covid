import logging as log
import os
from datetime import date

import keras.models as km
import pandas as pd
from tensorflow.python.keras.models import Model

import df_loader
import ml_transformer
from constants import *
from oxford_loader import df_geos

log.basicConfig(filename='predict.log', level=log.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')


def predict(start_date_str: str, end_date_str: str, path_future_data: str, path_output_file: str) -> None:
    """ Check the path, since the instructions are not really clear how bootstrap.sh is called """
    log.info(f"working directory is: {os.getcwd()}")
    if os.getcwd() == '/home/xprize':
        log.info('changing working directory to /home/xprize/work')
        os.chdir('/home/xprize/work')

    """  Log the input data """
    log.info(f"predicting {start_date_str} - {end_date_str} for '{path_future_data}' with output '{path_output_file}'")
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()

    """ 
    Load the dataframe that contains all historic and current data, with placeholders for future data.
    Then, initialize the structures we use to store the previous days new cases and confirmed case.
    The values are initialized to those from the last known date in the reference data:
        
        Confirmed Cases: for each geo, we search within the date range of our reference data, looking for the 
        MOST RECENT non-zero value, since we assume there may missing/corrupt data in the data set
        
        Predicted New Cases: this is calculated using a 'diff' from the dataset. 
        This is calculated when the data is loaded
    """
    df = df_loader.load_prediction_data(path_future_data, end_date)
    new_cases = {}
    confirmed_cases = {}

    for _, geo in df_geos.iterrows():
        geo_id = geo[GEO_ID]
        result = df[(df[GEO_ID] == geo_id) &
                    (df[DATE] <= pd.to_datetime(DATE_SUBMISSION_CUTOFF)) &
                    # for an index of -1, the date will be the 22nd: there will be no confirmed cases
                    # for an index of -2, the date will be the 21st: there will be cases, but nothing to predict
                    # for an index of -3, the date will be the 20th: there will be cases, and a prediction
                    (df[CONFIRMED_CASES] > 0)].iloc[-3]
        if result is None:
            log.error(f"no reference data found for {geo_id}")
            confirmed_cases[geo_id] = 0
            new_cases[geo_id] = 0
        else:
            print(f"{geo_id}:  confirmed = [{result[CONFIRMED_CASES]}], new = [{result[PREDICTED_NEW_CASES]}]")
            confirmed_cases[geo_id] = result[CONFIRMED_CASES]
            new_cases[geo_id] = result[PREDICTED_NEW_CASES]

    # load our model
    model = load_model(PREDICTED_NEW_CASES)

    # predict away!
    df = df[df[DATE] >= pd.to_datetime(DATE_SUBMISSION_CUTOFF)]
    df = df.groupby(DATE).apply(
        lambda group: predict_day(
            model,
            group,
            new_cases,
            confirmed_cases)).sort_values([COUNTRY_NAME, REGION_NAME, DATE])
    df.info()

    # save it baby!
    write_predictions(start_date, end_date, df, path_output_file)


def predict_day(model: Model,
                df_group: pd.DataFrame,
                new_cases,
                confirmed_cases) -> pd.DataFrame:
    log.info(f"predicting for {df_group[DATE].iloc[0]}")

    # Apply the previous day's prediction and confirmed cases to the current day
    df_group[CONFIRMED_CASES] = df_group[GEO_ID].apply(lambda x: confirmed_cases[x] + new_cases[x])

    # Calculate the next predictions!, perform a batch transformation of the full day records
    df_transformed = ml_transformer.transform(df_group.copy())
    model_predictions = model.predict(df_transformed)

    # Apply the predicted values back on the dataset
    idx = 0
    for _, row in df_group.iterrows():  # we need to map the geo_id to the index in the predictions
        geo_id = row[GEO_ID]
        value = model_predictions[idx][0] * LABEL_SCALING
        new_cases[geo_id] = value
        confirmed_cases[geo_id] += value
        idx = idx + 1

    # Apply the predicted cases to the date
    df_group[PREDICTED_NEW_CASES] = df_group[GEO_ID].apply(lambda x: new_cases[x])

    return df_group


def load_model(model_name: str):
    try:
        log.info('START loading model')
        model = km.load_model(f"models/{model_name}", compile=True)
        model.summary()
        log.info('END   loading model')
        return model
    except OSError:
        log.warning("unable to load model from path 'model', attempting to load from path 'work/model'")
        model = km.load_model('work/model', compile=True)
        model.summary()
        return model


def write_predictions(start_date, end_date, df: pd.DataFrame, path_output_file: str) -> None:
    log.info('START writing predictions')
    # Filter final results by date
    mask = (df[DATE] >= pd.to_datetime(start_date)) & (df[DATE] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    # Filter by Output Columns
    df = df[OUTPUT_COLUMNS]
    df.to_csv(path_output_file, index=False)
    log.info('END  writing predictions')


def date_range(start_date: date, end_date: date):
    days = int((end_date - start_date).days) + 1
    for n in range(int(days)):
        yield start_date + datetime.timedelta(n)
