import os
import sys
from six.moves import urllib
import pandas as pd
import numpy as np
from hotel.constant import *
from hotel.logger import logging
from hotel.entity.config_entity import DataIngestionConfig
from hotel.entity.artifact_entity import DataIngestionArtifact
from hotel.config.configuration import Configuartion
from hotel.exception import HotelException
from hotel.util.util import read_yaml_file
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*30}Data Ingestion log started.{'<<'*30} \n\n")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HotelException(e, sys) from e

    def download_data(self) -> str:
        try:
            download_url = self.data_ingestion_config.dataset_download_url

            raw_data_dir = self.data_ingestion_config.raw_data_dir

            os.makedirs(raw_data_dir, exist_ok=True)

            hotel_file_name = os.path.basename(download_url)

            raw_file_path = os.path.join(raw_data_dir, hotel_file_name)

            logging.info(
                f"Downloading file from :[{download_url}] into :[{raw_file_path}]")
            urllib.request.urlretrieve(download_url, raw_file_path)
            logging.info(
                f"File :[{raw_file_path}] has been downloaded successfully.")
            return raw_file_path

        except Exception as e:
            raise HotelException(e, sys) from e
    def _data_cleaning(self, df):
        try:
            # df["country"].fillna(df.country.mode()[0],inplace= True) 
            # df["agent"].fillna(0, inplace=True)
            df.drop(["company","name","email","phone-number","credit_card","company"], inplace= True, axis=1)
            zeroguest = (df["children"]+df["adults"]+df["babies"]==0)
            df.drop(df[zeroguest].index, inplace= True)
            df["lead_time"]= (df["lead_time"]/24).round(2)
            df["total_guests"]= df["children"]+df["adults"]+df["babies"]
            df.drop(["babies","adults","children"],axis=1,inplace=True)
            df["lead_time"]= (df["lead_time"]/24).round(2)

            df['hotel'] = df['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
            df['arrival_date_month'] = df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
            df['reserved_room_type'] = df['reserved_room_type'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'L': 8})
            
            # replace meal Undefined with Self Catering 
            df["meal"].replace("Undefined", "SC", inplace=True)
            # Replace 
            df["market_segment"].replace("Undefined", "Online TA", inplace=True)
            df.drop(df[df['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)

            df.drop(columns=['arrival_date_week_number',"reservation_status","reservation_status_date",
                    "assigned_room_type",'agent','required_car_parking_spaces', 'is_repeated_guest'], inplace=True, axis=1)
            df.drop(['previous_bookings_not_canceled', 'days_in_waiting_list'], inplace=True, axis=1)
            df[['lead_time']] = df[['lead_time']].apply(np.log1p)

            df.dropna(inplace=True)

            country= df['country'].value_counts().head(20)
            for i in range(df.shape[0]):
                if df['country'].iloc[i] in country:
                    continue
                else:
                    df['country'].iloc[i] = 'Others'

            return df
        except Exception as e:
            raise HotelException(e, sys) from e
        
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            file_name = os.listdir(raw_data_dir)[0]

            hotel_file_path = os.path.join(raw_data_dir, file_name)

            logging.info(f"Reading csv file: [{hotel_file_path}]")
            hotel_dataframe = pd.read_csv(hotel_file_path)
            
            hotel_dataframe = self._data_cleaning(hotel_dataframe)
            
            logging.info(f"Splitting data into train and test")

            train_set = None
            test_set = None

            train_set, test_set = train_test_split(hotel_dataframe, test_size=0.2, random_state=42)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                           file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                          file_name)

            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                train_set.to_csv(train_file_path, index=False)

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
            test_file_path=test_file_path,
            is_ingested=True,
            message=f"Data ingestion completed successfully."
            )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")

            return data_ingestion_artifact

        except Exception as e:
            raise HotelException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            raw_file_path = self.download_data()
            return self.split_data_as_train_test()
        except Exception as e:
            raise HotelException(e, sys)from e