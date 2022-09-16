import streamlit as st
import pickle
import time
import pandas as pd
from hotel.exception import HotelException
import os
import sys
import datetime
from hotel.util.util import get_country, date_extract
from hotel.logger import logging
from hotel.entity.hotel_predictor import HotelData, predictor
import pickle

countrylist = pickle.load(open("countryname.pkl", 'rb'))
# countrylist = get_country()


ROOT = os.getcwd()
SAVED_MODEL_DIR_NAME = "saved_models"
MODEL_DIR = os.path.join(ROOT,SAVED_MODEL_DIR_NAME)

@st.cache(suppress_st_warning=True, persist=True)

def main():
    try:
        logging.info("Build Streamlit app")
        st.sidebar.header(
            "Predict Hotel Booking Cancellations Using Machine Learning")
        st.sidebar.text("Choose the Parameters to Predict")

        st.sidebar.markdown("#### Hotel Type")
        hotel = st.sidebar.selectbox("Choose Hotel Type", [0, 1])
        st.sidebar.info("Resort: 1 City : 0")

        st.sidebar.markdown("#### Lead Time")
        lead_time = st.sidebar.slider(
            "Choose the Lead Time (Days)", 1, 4, step=1)

        st.sidebar.markdown("#### Country")
        country = st.sidebar.selectbox("Choose your Country", countrylist)

        st.sidebar.markdown("#### Arrival Date")
        arrival_date = st.sidebar.date_input(
            "When's your arrival date", datetime.date(2016, 2, 1))
        st.sidebar.info("Choose Between 2015-2017 only")

        st.sidebar.markdown("#### Weekend Stays")
        stays_in_weekend_nights = st.sidebar.slider(
            "Choose Weekend Night stays", 0, 10, step=1)

        st.sidebar.markdown("#### Weekdays Stays")
        stays_in_week_nights = st.sidebar.slider(
            "Choose Weekdays Night stays", 0, 6, step=1)

        st.sidebar.markdown("#### Total Guests")
        total_guests = st.sidebar.slider("Total Guests", 0, 50, step=1)

        st.sidebar.markdown("#### Average Daily Rate (ADR)")
        adr = st.sidebar.slider("Choose the ADR", 0, 210, step=25)

        st.sidebar.markdown("#### Meal Preference")
        meal = st.sidebar.selectbox("Choose your Meal Preference", [
                                    'BB', 'HB', 'FB', 'SC'])

        st.sidebar.markdown("#### Total Special Requests")
        total_of_special_requests = st.sidebar.slider(
            "Choose the number of special requests from guests", 0, 5, step=1)

        st.sidebar.markdown("#### Total Modifications")
        booking_changes = st.sidebar.slider(
            "Choose the number of modifications made by guests", 0, 14, step=1)

        st.sidebar.markdown("#### Previous Cancellations By Guest")
        previous_cancellations = st.sidebar.slider(
            "Choose the number of previous cancellations made by guests", 0, 25, step=1)

        st.sidebar.markdown("#### Market Segment")
        market_segment = st.sidebar.selectbox("Choose the Market Segment", ['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate','Complementary', 'Aviation'])
        st.sidebar.info("Through Which Platform you are Booking?")

        st.sidebar.markdown("#### Deposit Type")
        deposit_type = st.sidebar.selectbox("Choose the Deposit Type", [
                                            'No Deposit', 'Non Refund', 'Refundable'])

        st.sidebar.markdown("#### Reserved Room Type")
        reserved_room_type = st.sidebar.selectbox(
            "Choose the Reserved Room Type", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        st.sidebar.info("A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, L: 8")

        st.sidebar.markdown("#### Customer Type")
        customer_type = st.sidebar.selectbox("Choose the Customer Type", [
                                             'Transient', 'Transient-Party', 'Contract', 'Group'])

        st.sidebar.markdown("#### Distribution Type")
        distribution_channel = st.sidebar.selectbox("Choose the Distribution Type", [
                                                    'TA/TO', 'Direct', 'Corporate', 'GDS'])

        html_temp = """
            <div style="background-color:#000000 ;padding:10px">
            <h2 style="color:white;text-align:center;">Hotel Churn Rate Prediction Prediction</h2>
            </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        no_cancel_html = """
        <img src="https://i.gifer.com/55tE.gif" alt="confirmed" style="width:698px;height:350px;">
        """

        cancel_html = """
        <img src="https://i.gifer.com/5qV.gif" alt="cancel" style="width:698px;height:350px;">
        """

        hotelvideo_html = """
        <img src="https://media.giphy.com/media/f9SIUWTkLGOsQ9XaSX/giphy.gif" alt="success" style="width:698px;height:350px;">
        """
        st.markdown(hotelvideo_html, unsafe_allow_html=True)

        arrival_date_day_of_month, arrival_date_month, arrival_date_year = date_extract(date=str(arrival_date))


        hotel_data = HotelData(
            hotel=hotel,
            lead_time= lead_time,
            arrival_date_year= arrival_date_year,
            arrival_date_month= arrival_date_month,
            arrival_date_day_of_month= arrival_date_day_of_month,
            stays_in_weekend_nights= stays_in_weekend_nights,
            stays_in_week_nights= stays_in_week_nights,
            meal= meal,
            country= country,
            market_segment= market_segment,
            distribution_channel= distribution_channel,
            previous_cancellations= previous_cancellations,
            reserved_room_type= reserved_room_type,
            booking_changes= booking_changes,
            deposit_type= deposit_type,
            customer_type= customer_type,
            adr= adr,
            total_of_special_requests= total_of_special_requests,
            total_guests= total_guests
        )


        df = hotel_data.get_hotel_input_data_frame()
    except Exception as e:
        logging.error("Error in application code")
        raise HotelException(e, sys) from e



    # try:
    #     hotel_predictor = predictor(model_dir= MODEL_DIR)
    #     predicted_value = hotel_predictor.predict(X= df)[0]

    #     predicted_probability = hotel_predictor.proba_predict(X=df)[0][1]

    # except Exception as e:
    #     raise HotelException(e, sys) from e


    col1, col2, col3 = st.columns(3)
    if col2.button("Click Here To Predict"):
        hotel_predictor = predictor(model_dir= MODEL_DIR)
        predicted_value = hotel_predictor.predict(X= df)[0]
        predicted_probability = hotel_predictor.proba_predict(X=df)[0][1]

        logging.info("Output of the RUN is: {}".format(predicted_value))
        final_output=predicted_probability * 100
        st.header(
            'Chances of Guest Cancelling Reservation is {}% '.format(final_output))

        if final_output > 50.0:
            st.error("Reservation is not confirmed")
            st.markdown(cancel_html, unsafe_allow_html=True)
            logging.info("Output : Booking cancelled")
        else:
            st.balloons()
            time.sleep(2)
            st.balloons()
            st.success("Reservation is confirmed")
            st.markdown(no_cancel_html, unsafe_allow_html=True)
            logging.info("Output : Booking is confirmed")


if __name__ == '__main__':
    main()
