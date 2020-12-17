import datetime
import predict_io

WINDOW_LENGTH = 100  # this is the maximum amount of data we keep in the df, that is computable
START_DATE = datetime.date(2020, 11, 20)
END_DATE = datetime.date(2021, 1, 2)

# build our baseline window
df_baseline = predict_io.load_data_baseline()
window_start_date = df_baseline['date'].max()
print(window_start_date)
