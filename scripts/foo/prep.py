from pandas import DataFrame
import pandas as pd
from datetime import datetime, timedelta


def remove_spaces(path, ext):
    rd = pd.read_csv(path + '.' + ext)
    print(rd.columns)
    rd.columns = [((x.replace(" - ", "_")).replace(" ", "_")).lower()
                  for x in rd.columns]
    rd.to_csv(path + '.' + ext)


def build_dataset(path, ext):
    remove_spaces(path, ext)

    rd = pd.read_csv(path + '.' + ext,
                     parse_dates={'datetime': ['date_time']},
                     date_parser=lambda x:
                     datetime.strptime(x, '%d/%m/%Y %H:%M'))

    dates = rd['datetime']
    data = rd.drop('datetime', 1).replace(',', '.', regex=True).astype('float')

    rd = data.set_index(pd.DatetimeIndex(dates))
    rd.to_pickle(path + '.pickle')

    end = datetime(2014, 3, 31)
    start = end - timedelta(days=7)

    train = rd[:start-timedelta(seconds=1)]
    train.to_pickle(path + '_train.pickle')

    test = rd[start:end]
    test.to_pickle(path + '_test.pickle')

    return train, test

def build_dataset_v2(path, ext):



def transform_data(df, name):
    l1 = df[name].shift(1)
    l2 = df[name].shift(2)
    return DataFrame({'t': df[name], 'l1': l1, 'l2': l2}).dropna()
