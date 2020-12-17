import logging as log

import keras.models as km
import pandas as pd
import covid_constants

import predict_util
from loader import oxford_processor, oxford_loader
from pipeline import df_pipeline


def load_data_baseline() -> pd.DataFrame:
    log.info('START loading baseline data')
    df_baseline = df_pipeline.process_for_prediction(covid_constants.path_data_baseline)
    log.info('END   loading baseline data')
    return df_baseline


def load_future_data(path_future_data) -> pd.DataFrame:
    log.info('START loading future data')
    df_future = oxford_processor.process(oxford_loader.load(path_future_data))
    log.info(df_future.keys().values)
    log.info('END   loading future data')
    return df_future


def prepare_df_output(start_date, end_date) -> pd.DataFrame:
    log.info('START prepare_df_output')
    # Get the country and region data
    df_cr = pd.read_csv(covid_constants.reference_countries_and_regions)
    df_cr['RegionName'].fillna('', inplace=True)

    # Generate an array with dates for each country/region
    c = []
    r = []
    d = []
    for index, row in df_cr.iterrows():
        for date in predict_util.date_range(start_date, end_date, True):
            c.append(row['CountryName'])
            r.append(row['RegionName'])
            d.append(date)

    # Build the dataframe
    df_output = pd.DataFrame({'CountryName': c, 'RegionName': r, 'Date': d})
    df_output['PredictedDailyNewCases'] = 0.0
    df_output['IsSpecialty'] = 0

    log.info('END   prepare_df_output')
    return df_output


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


def write_predictions(df: pd.DataFrame, path_output_file: str) -> None:
    log.info('START writing predictions')
    df.to_csv(path_output_file, index=False)
    log.info('END  writing predictions')
