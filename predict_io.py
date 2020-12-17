import logging as log
from datetime import date

import keras.models as km
import pandas as pd
import datasets_constants

import predict_util
from loader import oxford_processor, oxford_loader
from pipeline import df_pipeline
from oxford_constants import *


def load_data_baseline() -> pd.DataFrame:
    log.info('START loading baseline data')
    df_baseline = df_pipeline.process_for_prediction(datasets_constants.PATH_DATA_BASELINE)
    log.info('END   loading baseline data')
    return df_baseline


def load_future_data(path_future_data) -> pd.DataFrame:
    log.info('START loading future data')
    df_future = oxford_processor.process(oxford_loader.load(path_future_data))
    log.info(df_future.keys().values)
    log.info('END   loading future data')
    return df_future


def load_model():
    try:
        log.info('START loading model')
        model = km.load_model('model', compile=True)
        model.summary()
        log.info('END   loading model')
        return model
    except OSError:
        log.warning("unable to load model from path 'model', attempting to load from path 'work/model'")
        model = km.load_model('work/model', compile=True)
        model.summary()
        return model


"""
 Compute a data for predictions
 
 Some important considerations in processing the data:
 
 1. when there is a gap between the last day in the historical data and the start of the prediction date,
    we must include enough historical data to project ahead. We also must consider that

 2. Not all countries / regions may have historical data. For this case, we set it to zero for now

"""


def prepare_df_window(start_date: date, end_date: date) -> pd.DataFrame:
    # Get the baseline data, determine the max date, and set the initial window to be used
    df_baseline = load_data_baseline()
    window_start_date = df_baseline[DATE].max()

    # Get the country and region data
    df_cr = pd.read_csv(datasets_constants.REFERENCE_COUNTRIES_AND_REGIONS)
    df_cr[REGION_NAME].fillna('', inplace=True)

    return None


# TODO REMOVE THIS FUNCTION
def prepare_df_output(start_date: date, end_date: date) -> pd.DataFrame:
    log.info('START prepare_df_output')
    # Get the country and region data
    df_cr = pd.read_csv(datasets_constants.REFERENCE_COUNTRIES_AND_REGIONS)
    df_cr[REGION_NAME].fillna('', inplace=True)

    # Generate an array with dates for each country/region
    col_country = []
    col_region = []
    col_date = []
    for index, row in df_cr.iterrows():
        for d in predict_util.date_range(start_date, end_date, True):
            col_country.append(row[COUNTRY_NAME])
            col_region.append(row[REGION_NAME])
            col_date.append(d)

    # Build the dataframe
    df_output = pd.DataFrame({COUNTRY_NAME: col_country, REGION_NAME: col_region, DATE: col_date})
    df_output[PREDICTED_NEW_CASES] = 0.0
    df_output[IS_SPECIALTY] = 0
    log.info('END   prepare_df_output')
    return df_output


# TODO: add date constraints for the output
def write_predictions(df: pd.DataFrame, path_output_file: str) -> None:
    # TODO: restrict output to specific start dates, countries, and regions here...
    log.info('START writing predictions')
    df.to_csv(path_output_file, index=False)
    log.info('END  writing predictions')
