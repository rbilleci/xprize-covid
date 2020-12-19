import logging as log
import os
from datetime import date

import keras.models as km
import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
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

    """ Load the unique geo ids will handle """

    """ Load the dataframe that contains all historic and current data, with placeholders for future data"""
    df = df_loader.load_prediction_data(path_future_data, end_date)

    """ Load the model """
    model = load_model(PREDICTED_NEW_CASES)

    """ Iterate over each day """
    grouped = df.groupby([DATE])
    predictions = {}
    for day, df_day in grouped:
        predict_day(model, day, df_day, predictions)
    # TODO: ungroup?

    """ Write the predictions """

    # TODO: !
    # write_predictions(start_date, end_date, df, path_output_file)


def predict_day(model: Model,
                day: Timestamp,
                df_group: pd.DataFrame,
                predictions: {}) -> None:
    print(f"START prediction {day}")
    # TODO: we need the correct start date, think backward to know when we'll have final values...
    # Assign the prediction from the previous day to the current
    for _, geo in df_geos.iterrows():
        geo_id = geo[GEO_ID]
        if geo_id in predictions:
            df_group[CONFIRMED_CASES][(df_group[GEO_ID == geo_id])] = df_group[CONFIRMED_CASES] + predictions[geo_id]
        predictions[geo_id] = None

    # Calculate the next predictions!
    for _, r in df_group.iterrows():
        df_x = pd.DataFrame.from_records([r])
        df_x = ml_transformer.transform(df_x)
        df_x = df_x.drop([PREDICTED_NEW_CASES, IS_SPECIALTY], axis=1)
        prediction = model.predict(np.array([df_x.iloc[0]]))[0][0] * LABEL_SCALING
        print(f"f{prediction} was predicted")
        predictions[r[GEO_ID]] = prediction


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
