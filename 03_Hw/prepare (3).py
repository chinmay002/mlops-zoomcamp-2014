from typing import Tuple

import pandas as pd

from mlops.utils.data_preparation.cleaning import clean


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(df):
    
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    print('aa')
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    print('bb')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime

    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    print('cc')
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df[['PULocationID', 'DOLocationID','duration']]

def test_output(df):
    assert df is not None