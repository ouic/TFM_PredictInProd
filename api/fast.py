from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
from TaxiFareModel.params import PATH_TO_LOCAL_MODEL
from predict import get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def X_predict(pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
    df={}
    df['key']= ['1']
    df['pickup_datetime']=[formatted_pickup_datetime]
    df['pickup_longitude']=[pickup_longitude]
    df['pickup_latitude']=[pickup_latitude]
    df['dropoff_longitude']=[dropoff_longitude]
    df['dropoff_latitude']=[dropoff_latitude]
    df['passenger_count']=[passenger_count]
    df= pd.DataFrame(df)
    model = get_model(PATH_TO_LOCAL_MODEL)
    prediction = model.predict(df)
    return prediction[0]