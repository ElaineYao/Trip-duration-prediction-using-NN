# Trip-duration-prediction-using-NN
This is a simple project to predict the approximate time during a taxi trip. 

The input includes the latitude & longitude of the pick-up & drop-off sites and the pick-up time.

The output predicted time is in the form of '0-5min', '5-10min', '10-15min', '15-20min', '20-15min', '25-30min, '>30min'

# Dataset
[New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

Here I just used the `train.zip` as the whole training and testing dataset.

I put the `train.csv` under the `dataset` directory. Please change the `datapath` in `utils.py` according to your path.

# Getting Started
`run train.py` 

# Data Preprocessing

