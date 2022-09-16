from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from hotel.entity.hotel_predictor import predictor, HeartStrokeData
from hotel.pipeline.pipeline import Pipeline
from hotel.config.configuration import Configuartion
from fastapi import Body
from typing import Dict, Any

import pandas as pd

app = FastAPI()

class HotelDataInput(BaseModel):
    hotel: float
    lead_time: float
    arrival_date_year: float
    arrival_date_month: float
    arrival_date_day_of_month: float
    stays_in_weekend_nights: float
    stays_in_week_nights: float
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    previous_cancellations: float
    reserved_room_type: float
    booking_changes: float
    deposit_type: str
    customer_type: str
    adr: float
    total_of_special_requests: float
    total_guests: float

@app.get("/train")
async def read_root():
    try:
        response = Pipeline(Configuartion()).run()
        return {"msg" : "Training Completed"}
    except Exception as e:
        raise e

@app.post("/predict")
async def read_item(data: HotelDataInput):
    hotel  = data['hotel']
    lead_time  = data['lead_time']
    arrival_date_year  = data['arrival_date_year']
    arrival_date_month  = data['arrival_date_month']
    arrival_date_day_of_month =data["arrival_date_day_of_monthll"]
    stays_in_weekend_nights =data["stays_in_weekend_nights"]
    stays_in_week_nights =data["stays_in_week_nights"]
    meal =data["meal"]
    country =data["country"]
    market_segment =data["market_segment"]
    distribution_channel =data["distribution_channel"]
    previous_cancellations =data["previous_cancellationsQuantity"]
    reserved_room_type =data["reserved_room_type"]
    booking_changes =data["booking_changes"]
    deposit_type =data["deposit_type"]
    customer_type =data["customer_type"]
    adr =data["adr"]
    total_of_special_requests =data["total_of_special_requestsoProcess"]
    total_guests =data["total_guests"]

    dataClass = HeartStrokeData(
        hotel,
        lead_time,
        arrival_date_year,
        arrival_date_month,
        arrival_date_day_of_month,
        stays_in_weekend_nights,
        stays_in_week_nights,
        meal,
        country,
        market_segment,
        distribution_channel,
        previous_cancellations,
        reserved_room_type,
        booking_changes,
        deposit_type,
        customer_type,
        adr,
        total_of_special_requests,
        total_guests
    )
    df = dataClass.get_heart_stroke_input_data_frame()
    # df = pd.read_csv(r'D:\Consignment\ConsignmentPricing\artifacts\2022-09-15-13-09-23\data_cleaning\cleaned_consignment_data.csv')
    # df = df.iloc[:1]

    predictorClass = predictor(model_dir=r"D:\Hotel_Booking_Predictor_System\saved_models\20220916121036\model.pkl")
    
    result = predictorClass.predict(df)
    return {"msg": "Prediction Completed",
        "result": result}