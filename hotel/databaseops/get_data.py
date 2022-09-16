from hotel.databaseops import mongodbconnection
from hotel.exception import HotelException
from hotel.logger import logging
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys


def get_training(username, password):
    try:
        dbcon = mongodbconnection(username, password)
        list_cursor = dbcon.getdata(dbName='Hotel_db', collectionName='train')
        logging.info('Connected to Mongodb and data retrieved')
    except Exception as e:
        raise HotelException(e, sys) from e

    # Data From MongoDB is used for Standardization
    df = pd.DataFrame(list_cursor)
    df = df.drop(columns='_id', axis=1)
    return df
