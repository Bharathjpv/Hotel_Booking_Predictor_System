import yaml
from hotel.exception import HotelException
import os, sys
import numpy as np
import dill
import pandas as pd
from hotel.constant import *
import datetime


def date_extract(date):
    datem = datetime.datetime.strptime(date, "%Y-%m-%d")
    return datem.day, datem.month, datem.year


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise HotelException(e,sys)


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HotelException(e,sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HotelException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HotelException(e, sys) from e


def save_object(file_path: str, obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise HotelException(e, sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise HotelException(e, sys) from e


def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        dataset_schema = read_yaml_file(schema_file_path)

        schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_message = ""

        # for column in dataframe.columns:
        #     if column in list(schema.keys()):
        #         dataframe[column].astype(schema[column])
        #     else:
        #         error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
        # if len(error_message) > 0:
        #     raise Exception(error_message)
        return dataframe

    except Exception as e:
        raise HotelException(e, sys) from e

def get_country():
    try:
        data_url ="https://raw.githubusercontent.com/Bharathjpv/Datasets/main/hotel_booking.csv"
        df =pd.read_csv(data_url)

        country= df['country'].value_counts().head(20)
        for i in range(df.shape[0]):
            if df['country'].iloc[i] in country:
                continue
            else:
                df['country'].iloc[i] = 'Others'
        country = list(df.country.unique())
        return country
    except Exception as e:
        raise CarException(e,sys) from e


    