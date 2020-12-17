import datetime
import logging as log
import predict_util
import keras.models as km

log.basicConfig(filename='predict.log', level=log.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')


def predict(start_date_str: str,
            end_date_str: str,
            fn_in: str,
            fn_out) -> None:
    # Perform sanity checks
    log.info(f"predict for {start_date_str} to {end_date_str} with file {fn_in} and output file {fn_out}")
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    log.info(f"parsed start date as: {start_date}")
    log.info(f"parsed end date as: {end_date}")
    log.info(f"reading input file {fn_in}")

    # get the model
    model = resolve_model()

    # run the predictions through each date
    for prediction_date in predict_util.date_range(start_date, end_date, True):
        predict_day(prediction_date)
    return None


def predict_day(prediction_date):
    log.info(f"evaluating predictions for date: {prediction_date}")


def resolve_model():
    try:
        model = km.load_model('model', compile=True)
        model.summary()
        return model
    except OSError:
        log.warning("unable to load model from path 'model', attempting to load from path 'work/model'")
        model = km.load_model('work/model', compile=True)
        model.summary()
        return model
