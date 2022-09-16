import os
import sys

from hotel.exception import HotelException
from hotel.util.util import load_object

import pandas as pd


class HotelData:

    def __init__(self,
                hotel: float,
                lead_time: float,
                arrival_date_year: float,
                arrival_date_month: float,
                arrival_date_day_of_month: float,
                stays_in_weekend_nights: float,
                stays_in_week_nights: float,
                meal: str,
                country: str,
                market_segment: str,
                distribution_channel: str,
                previous_cancellations: float,
                reserved_room_type: float,
                booking_changes: float,
                deposit_type: str,
                customer_type: str,
                adr: float,
                total_of_special_requests: float,
                total_guests: float,
                is_canceled: float = None,
                ):
        try:
            self.hotel = hotel
            self.lead_time = lead_time
            self.arrival_date_year = arrival_date_year
            self.arrival_date_month = arrival_date_month
            self.arrival_date_day_of_month = arrival_date_day_of_month
            self.stays_in_weekend_nights = stays_in_weekend_nights
            self.stays_in_week_nights = stays_in_week_nights
            self.meal = meal
            self.country = country
            self.market_segment = market_segment
            self.distribution_channel = distribution_channel
            self.previous_cancellations = previous_cancellations
            self.reserved_room_type = reserved_room_type
            self.booking_changes = booking_changes
            self.deposit_type = deposit_type
            self.customer_type = customer_type
            self.adr = adr
            self.total_of_special_requests = total_of_special_requests
            self.total_guests = total_guests

        except Exception as e:
            raise HotelException(e, sys) from e

    def get_hotel_input_data_frame(self):

        try:
            heart_stroke_input_dict = self.get_heart_stroke_data_as_dict()
            return pd.DataFrame(heart_stroke_input_dict)
        except Exception as e:
            raise HotelException(e, sys) from e

    def get_hotel_data_as_dict(self):
        try:
            input_data = {
                "hotel": [self.hotel],
                "lead_time": [self.lead_time],
                "arrival_date_year": [self.arrival_date_year],
                "arrival_date_month": [self.arrival_date_month],
                "arrival_date_day_of_month": [self.arrival_date_day_of_month],
                "stays_in_weekend_nights": [self.stays_in_weekend_nights],
                "stays_in_week_nights": [self.stays_in_week_nights],
                "meal": [self.meal],
                "country": [self.country],
                "market_segment": [self.market_segment],
                "distribution_channel": [self.distribution_channel],
                "previous_cancellations": [self.previous_cancellations],
                "reserved_room_type": [self.reserved_room_type],
                "booking_changes": [self.booking_changes],
                "deposit_type": [self.deposit_type],
                "customer_type": [self.customer_type],
                "adr": [self.adr],
                "total_of_special_requests": [self.total_of_special_requests],
                "total_guests": [self.total_guests]
                }
            return input_data
        except Exception as e:
            raise HotelException(e, sys)


class predictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise HotelException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise HotelException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            predited_value = model.predict(X)
            return predited_value
        except Exception as e:
            raise HotelException(e, sys) from e
        
    def proba_predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            probaility = model.predict_proba(X)
            return probaility 
        except Exception as e:
            raise HotelException(e, sys) from e