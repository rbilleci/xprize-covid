import pandas as pd

import df_10_data_timeinfo
import df_11_countryinfo
import df_00_splitter
import df_50_mark_nulls
import df_60_imputer
import df_80_scaler
import oxford_loader
import oxford_processor
import df_70_label
import df_90_ohe


# TODO: convert is working day to bloom filters to save loads of memory
# TODO: check that we are always on the right pipeline
# TODO: check if we can expect any of the cN values to be NULL!!!!!!
# TODO: sanity check for null dates!
# TOdO: country info
# TODO: working day info
# TODO: holiday info
def process_for_training(fn: str,
                         days_for_validation: int,
                         days_for_test: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = oxford_loader.load(fn)
    df = oxford_processor.process(df)
    train, validation, test = df_00_splitter.split(df, days_for_validation, days_for_test)
    return _process_split(train), _process_split(validation), _process_split(test)


def process_for_prediction(fn: str) -> pd.DataFrame:
    return _process_split(oxford_processor.process(oxford_loader.load(fn)))


def _process_split(df) -> pd.DataFrame:
    df = df_10_data_timeinfo.apply(df)
    df = df_11_countryinfo.apply(df)
    df = df_50_mark_nulls.apply(df)
    df = df_60_imputer.apply(df)
    df = df_70_label.apply(df) # apply label before scaling, so that it is not scaled twice
    df = df_80_scaler.apply(df)
    df = df_90_ohe.apply(df)
    # move label to first column
    label = df['_label']
    df.drop(labels=['_label'], axis=1, inplace=True)
    df.insert(0, '_label', label)
    return df
