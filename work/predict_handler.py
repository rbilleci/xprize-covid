import logging as log
import datetime


def predict(start_date_str: str,
            end_date_str: str,
            fn_in: str,
            fn_out) -> None:
    # configure logging
    log.basicConfig(filename='test.log', level=log.INFO, format='%(asctime)s %(filename)s: %(message)s')

    # perform sanity checks
    log.info(f"predict for {start_date_str} to {end_date_str} with file {fn_in} and output file {fn_out}")
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    log.info(f"parsed start date as: {start_date}")
    log.info(f"parsed end date as: {end_date}")
    log.info(f"reading input file {fn_in}")
    return None
