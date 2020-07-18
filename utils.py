import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

datapath = './dataset/'

# Make the trip_duration time discrete in minutes,
# input: the dataframe of train or test.csv
# return: dataframe with 2 columns. The first is id, the second is labeled trip_duration time
# Tip: the one-hot encoding will be done after all the data is label
def output_preprocess(input):
    df_output = pd.DataFrame(columns = ['id', 'trip_duration'])
    df_output.loc[:, 'id'] = input['id']
    df_output.loc[:, 'trip_duration'] = np.round(input['trip_duration']/60)
    df_output.loc[:, 'trip_duration'] = df_output['trip_duration'].map(output_label)
    return df_output

# Label the output.
# <5 min -> label 0; 5~10min -> label 1; 10~15min -> label 2; 15~20min -> label 3;
#  20~25min -> label 4;  25~30min -> label 5;  >30min -> label 6;
def output_label(trip_duration):
    if trip_duration<5:
        return 0
    elif (trip_duration>=5)&(trip_duration<10):
        return 1
    elif (trip_duration>=10)&(trip_duration<15):
        return 2
    elif (trip_duration>=15)&(trip_duration<20):
        return 3
    elif (trip_duration>=20)&(trip_duration<25):
        return 4
    elif (trip_duration>=25)&(trip_duration<30):
        return 5
    elif trip_duration>=30:
        return 6


# Define some distance features
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b



def get_zone_features(input_temp):
    # Define zone features
    # First，delete some outliers
    input_df = input_temp.loc[(input_temp.pickup_latitude > 40.6) & (input_temp.pickup_latitude < 40.9)]
    input_df = input_df.loc[(input_df.dropoff_latitude > 40.6) & (input_df.dropoff_latitude < 40.9)]
    input_df = input_df.loc[(input_df.dropoff_longitude > -74.05) & (input_df.dropoff_longitude < -73.7)]
    input_df = input_df.loc[(input_df.pickup_longitude > -74.05) & (input_df.pickup_longitude < -73.7)]

    # Then create the lat_long grid
    pick_lat_max = input_df.loc[:, 'pickup_latitude'].max()
    drop_lat_max = input_df.loc[:, 'dropoff_latitude'].max()
    lat_max = max(pick_lat_max, drop_lat_max)

    pick_lat_min = input_df.loc[:, 'pickup_latitude'].min()
    drop_lat_min = input_df.loc[:, 'dropoff_latitude'].min()
    lat_min = min(pick_lat_min, drop_lat_min)

    pick_long_max = input_df.loc[:, 'pickup_longitude'].max()
    drop_long_max = input_df.loc[:, 'dropoff_longitude'].max()
    long_max = max(pick_long_max, drop_long_max)

    pick_long_min = input_df.loc[:, 'pickup_longitude'].min()
    drop_long_min = input_df.loc[:, 'dropoff_longitude'].min()
    long_min = min(pick_long_min, drop_long_min)

    print(lat_max, lat_min, long_max, long_min)
    return input_df, lat_min, long_min

#
def get_standard_onehot_x(x_data, input_df):
    x_data.loc[:, 'id'] = input_df.id
    x_data.loc[:, 'vendor_id'] = input_df.vendor_id
    x_data.loc[:, 'distance_haversine'] = input_df.distance_haversine
    x_data.loc[:, 'distance_dummy_manhattan'] = input_df.distance_dummy_manhattan
    x_data.loc[:, 'avg_speed_h'] = input_df.avg_speed_h
    x_data.loc[:, 'avg_speed_m'] = input_df.avg_speed_m
    x_data.loc[:, 'pickup_weekday'] = input_df.pickup_weekday
    x_data.loc[:, 'pickup_hour'] = input_df.pickup_hour
    x_data.loc[:, 'pickup_minute'] = input_df.pickup_minute
    x_data.loc[:, 'pickup_lat_label'] = input_df.pickup_lat_label
    x_data.loc[:, 'pickup_long_label'] = input_df.pickup_long_label
    x_data.loc[:, 'dropoff_lat_label'] = input_df.dropoff_lat_label
    x_data.loc[:, 'dropoff_long_label'] = input_df.dropoff_long_label

    # Drop id
    x_data = x_data.drop('id', axis=1)

    # Standardize some features -- 'distance_haversine', 'distance_dummy_manhattan', 'avg_speed_h', 'avg_speed_m'
    x_data[['distance_haversine', 'distance_dummy_manhattan', 'avg_speed_h',
            'avg_speed_m']] = StandardScaler().fit_transform(x_data[['distance_havers'
                                                                     'ine', 'distance_dummy_manhattan', 'avg_speed_h',
                                                                     'avg_speed_m']])
    # One-hot encode the other features
    x_data_array = x_data.values
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [5, 6, 7, 8, 9, 10, 11])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )
    x_data_onehot = ct.fit_transform(x_data_array).toarray()
    return x_data_onehot