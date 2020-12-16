import df_pipeline
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_info_columns = 1000

print("preparing training data")
train, validation, test = df_pipeline.process_for_training('OxCGRT_latest.csv', 21, 21)

print("preparing historical data")
df = df_pipeline.process_for_prediction('2020-09-30_historical_ip.csv')

print("preparing future data")
df = df_pipeline.process_for_prediction('future_ip.csv')
