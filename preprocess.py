from utils import *
def preprocess(datapath):
    input_temp = pd.read_csv(os.path.join(datapath, 'train.csv'))

    # Define input features
    # Distance features: 'distance_haversine', 'distance_dummy_manhattan'
    # Speed features：'avg_speed_h'，'avg_speed_m', unit:m/s
    # Time features: 'pick_up_h', 'pick_up_m', 'weekday', (0 represents Sunday)
    # Zone features: 'pickup_lat_label', 'pickup_long_label', 'dropoff_lat_label', 'dropoff_long_label'

    # Define distance features
    input_temp.loc[:, 'distance_haversine'] = haversine_array(input_temp['pickup_latitude'].values,
                                                              input_temp['pickup_longitude'].values,
                                                              input_temp['dropoff_latitude'].values,
                                                              input_temp['dropoff_longitude'].values)
    input_temp.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(input_temp['pickup_latitude'].values,
                                                                             input_temp['pickup_longitude'].values,
                                                                             input_temp['dropoff_latitude'].values,
                                                                             input_temp['dropoff_longitude'].values)
    # Define speed features
    input_temp.loc[:, 'avg_speed_h'] = 1000 * input_temp['distance_haversine'] / input_temp['trip_duration']
    input_temp.loc[:, 'avg_speed_m'] = 1000 * input_temp['distance_dummy_manhattan'] / input_temp['trip_duration']

    # Define time features
    input_temp['pickup_datetime'] = pd.to_datetime(input_temp.pickup_datetime)
    input_temp.loc[:, 'pickup_weekday'] = input_temp['pickup_datetime'].dt.weekday
    input_temp.loc[:, 'pickup_hour'] = input_temp['pickup_datetime'].dt.hour
    input_temp.loc[:, 'pickup_minute'] = input_temp['pickup_datetime'].dt.minute

    # Define zone features
    input_df, lat_min, long_min = get_zone_features(input_temp)
    input_df.loc[:, 'pickup_lat_label'] = np.round((input_df.pickup_latitude - lat_min) / 0.01)
    input_df.loc[:, 'pickup_long_label'] = np.round((input_df.pickup_longitude - long_min) / 0.01)
    input_df.loc[:, 'dropoff_lat_label'] = np.round((input_df.dropoff_latitude - lat_min) / 0.01)
    input_df.loc[:, 'dropoff_long_label'] = np.round((input_df.dropoff_longitude - long_min) / 0.01)

    # Standardize and one-hot encode the input data
    x_data = pd.DataFrame(
        columns=['id', 'vendor_id', 'distance_haversine', 'distance_dummy_manhattan', 'avg_speed_h', 'avg_speed_m',
                 'pickup_weekday', 'pickup_hour', 'pickup_minute', 'pickup_lat_label', 'pickup_long_label',
                 'dropoff_lat_label', 'dropoff_long_label'])
    x_data_onehot = get_standard_onehot_x(x_data, input_df)

    # One-hot encode the output data
    y_data = output_preprocess(input_df)
    y_data = y_data.drop('id', axis=1)
    y_data_array = y_data.values
    y_data_reshape = y_data_array.flatten()
    y_data_onehot = to_categorical(y_data_reshape)

    # Split the data into training set, test set and validation set
    x_train_tmp, x_test, y_train_tmp, y_test = train_test_split(x_data_onehot, y_data_onehot, test_size=0.2,
                                                                random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_tmp, y_train_tmp, test_size  = 0.2, random_state=55)

    return x_train, y_train, x_val, y_val, x_test, y_test