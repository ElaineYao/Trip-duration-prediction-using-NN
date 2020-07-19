from utils import *
from preprocess import *
from network import *
from train import *
from keras.models import load_model
from sklearn import preprocessing

def run():
    # Run the model
    model_path = 'model1_10epoch.h5'
    my_model = load_model(model_path)
    predict_int = my_model.predict_classes(test_X).astype('int')
    predict_int_list = predict_int.tolist()
    # Convert to the form of the duration time
    dic = {0:'0-5min', 1:'5-10min', 2:'10-15min', 3:'15-20min', 4:'20-25min', 5:'25-30min', 6:'>30min'}
    predicted_label=[dic[i] if i in dic else i for i in predict_int_list]
    print('The predicted trip duration is around {}'.format(predicted_label))

if __name__ == '__main__':
    # Define the input
    # You can change the input
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) = preprocess(datapath)
    run()