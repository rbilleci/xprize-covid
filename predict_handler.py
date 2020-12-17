import datetime
import logging as log
import predict_util
import keras.models as km
from pipeline import df_pipeline

log.basicConfig(filename='predict.log', level=log.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')


def predict(start_date_str: str,
            end_date_str: str,
            path_future_data: str,
            path_output_file: str) -> None:
    # Perform sanity checks
    log.info(f"predicting {start_date_str} - {end_date_str} for '{path_future_data}' with output '{path_output_file}'")
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    log.info(f"parsed start date as: {start_date}")
    log.info(f"parsed end date as: {end_date}")

    # get the baseline data we'll predict from
    df_baseline = load_baseline_data()
    df_future = load_future_data(path_future_data)
    model = load_model()

    # run the predictions through each date
    for prediction_date in predict_util.date_range(start_date, end_date, True):
        predict_day(prediction_date)
    return None


def predict_day(prediction_date):
    log.info(f"START prediction {prediction_date}")
    log.info(f"END   prediction {prediction_date}")


def load_baseline_data():
    log.info('START loading baseline data')
    df_baseline = df_pipeline.process_for_prediction('data/baseline_data.csv')
    log.info('END   loading baseline data')
    df_baseline.info()
    return df_baseline


def load_future_data(path_future_data):
    log.info('START loading baseline data')
    df_baseline = df_pipeline.process_for_prediction(path_future_data)
    log.info('END   loading baseline data')
    df_baseline.info()
    return df_baseline


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
