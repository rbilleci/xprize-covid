import datetime
import logging as log
import os
from datetime import date

import keras.models as km
import pandas as pd

from oxford_constants import DATE, OUTPUT_COLUMNS, PREDICTED_NEW_CASES
from pipeline import df_pipeline

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

    """ Load the dataframe that contains all historic and current data, with placeholders for future data"""
    df = df_pipeline.get_dataset_for_prediction(start_date, end_date, path_future_data)

    """ Load the model """
    model = load_model(PREDICTED_NEW_CASES)

    """ Iterate over each day """
    grouped = df.groupby(DATE)
    grouped = grouped.apply(lambda group: predict_day(model, group))
    # df = grouped.reset_index(drop=True)
    # for prediction_date in date_range(start_date, end_date):  # TODO: need to fix the start date???
    #    predict_day(model, prediction_date)

    """ Write the predictions """
    # write_predictions(start_date, end_date, df, path_output_file)


def predict_day(model_cases, group: pd.DataFrame) -> None:
    print(f"START prediction {group[DATE].min()} - {group[DATE].max()}")
    # log.info(f"START prediction {type(group)}")


def predict_country(model_cases) -> None:
    return None


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
