import datetime
import logging as log

import predict_io
import predict_util

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
    df_baseline = predict_io.load_data_baseline()
    df_future = predict_io.load_future_data(path_future_data)
    model = predict_io.load_model()
    df_output = predict_io.prepare_df_output(start_date, end_date)

    # run the predictions through each date
    for prediction_date in predict_util.date_range(start_date, end_date, True):
        predict_day(prediction_date)

    # write the predictions
    predict_io.write_predictions(df_output, path_output_file)
    return None


def predict_day(prediction_date):
    log.info(f"START prediction {prediction_date}")
    log.info(f"END   prediction {prediction_date}")
