# Trip-duration-prediction-using-NN
This is a simple project to predict the approximate time during a taxi trip. 

The input includes the latitude & longitude of the pick-up & drop-off sites and the pick-up time.

The output predicted time is in the form of '0-5min', '5-10min', '10-15min', '15-20min', '20-15min', '25-30min, '>30min'

# How to run
To run this project, you can follow the steps below.

### To run predictions using the pre-trained model:
The model parameters have be saved in `model1_10epoch.h5` file. 

I've set the input to be the original test data, so just `run run.py` .

### Training the model from scratch
- Download the [New York City Taxi Trip Duration Dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) and store it in a directory `dataset`. Here I just used the `train.zip` as the whole training and testing dataset. Please change the `datapath`(the path for train.csv) in `utils.py` according to your path.
- `run train.py`
- `run run.py`

# Data Preprocessing


