import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


def _remove_spaces(df):
    df.columns = [((x.replace(" - ", "_")).replace(" ", "_")).lower()
                  for x in df.columns]
    return df


def _get_names(df, z):
    return list(filter(lambda x: any(map(lambda y: y in x, z)),
                       df.columns.values))

def _replace_comma(x):
    if x != np.nan:
        x = str(x).replace(',', '.')

    return x

def _date_parser(x):
    return pd.datetime.strptime(x, '%d/%m/%Y %H:%M')


def prep_dataframe(keep=None, drop=None, remove_mean=False, normalise=False):
    room_level = os.path.join(os.path.dirname(__file__), "../data/room_level.csv")
    racks_temps = os.path.join(os.path.dirname(__file__), "../data/rack_temps.csv")
    ahus = os.path.join(os.path.dirname(__file__), "../data/ahus.csv")

    df_room = pd.read_csv(room_level, sep=';', parse_dates=['date_time'], date_parser=_date_parser, decimal=',')

    df_rack_temps = pd.read_csv(racks_temps, sep=';', parse_dates=['date_time'], date_parser=_date_parser, decimal=',')

    df_ahus = pd.read_csv(ahus, sep=';', parse_dates=['date_time'], date_parser=_date_parser, decimal=',')

    df = df_rack_temps.merge(df_ahus, on="date_time", how="outer")
    df = df.merge(df_room, on="date_time", how="outer")
    df = df.sort('date_time')
    df = df.set_index('date_time')

    df = _remove_spaces(df)

    if keep:
        df = df[_get_names(df, keep)]

    if drop:
        df = df.drop(_get_names(df, drop), 1)

    df = df.interpolate(method="time")
    df = df.dropna()

    if remove_mean:
        df = df - df.mean()

    if normalise == True:
        values = preprocessing.scale(df.values)
        index = df.index
        columns = df.columns
        df = pd.DataFrame.from_records(values, index=index, columns=columns)

    return df


def create_lagged_features(df, cols=None):
    if not cols:
        cols = df.columns

    df_lagged = pd.DataFrame(index=df.index)
    for c in cols:
        df_lagged[c] = df[c] - df[c].shift(1)

    df_lagged = df_lagged.dropna()

    return df_lagged

def create_shifted_features(df, shift=1, cols=None):
    if not cols:
        cols = df.columns

    df_shifted = pd.DataFrame(index=df.index)
    for c in cols:
        df_shifted["l" + str(shift) + "_" + c] = df[c].shift(shift)


    return df_shifted
