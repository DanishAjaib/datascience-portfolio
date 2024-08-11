import joblib
import pandas as pd
import modelbit as mb

_model = joblib.load("models/british_airways/model.pkl")


def predict_booking_complete(
    num_passengers,
    sales_channel,
    trip_type,
    purchase_lead,
    length_of_stay,
    flight_hour,
    flight_day,
    route,
    booking_origin,
    wants_extra_baggage,
    wants_preferred_seat,
    wants_in_flight_meals,
    flight_duration):
    

    # Create a dataframe with the new data
    new_data = pd.DataFrame({
        'num_passengers': [num_passengers],
        'sales_channel': [sales_channel],
        'trip_type': [trip_type],
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour],
        'flight_day': [flight_day],
        'route': [route],
        'booking_origin': [booking_origin],
        'wants_extra_baggage': [wants_extra_baggage],
        'wants_preferred_seat': [wants_preferred_seat],
        'wants_in_flight_meals': [wants_in_flight_meals],
        'flight_duration': [flight_duration]
    })

    
    # Make a prediction
    probablity = _model.predict_proba(new_data)
    prediction = _model.predict(new_data)
    return prediction, probablity

mb.deploy(
    predict_booking_complete,             # The function to deploy
    name="british_airways",                  # The name of the model or service
    python_packages=["requirements.txt"]  ,
       # List of required packages
)