import pandas as pd


def scale_value(x, min_value, max_value, range_is_zero_to_one=True):
    x = (x - min_value) / (max_value - min_value)
    if range_is_zero_to_one:
        x = max(0.0, x)
    else:
        x = max(-1.0, x)
    x = min(1.0, x)
    return x


def scale(df, name, min_value, max_value, range_is_zero_to_one=True):
    df[f"{name}_scaled"] = df[name].apply(lambda x: scale_value(x, min_value, max_value, range_is_zero_to_one))
    df.drop(name, axis=1, inplace=True)
