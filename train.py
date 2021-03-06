from utils import *
from preprocess import *
from network import *

def train():
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) = preprocess(datapath)
    model = network()
    history = model.fit(x=train_X, y=train_Y, validation_data = (val_X, val_Y),
                    batch_size=64,
                      epochs=10)
    # Calculate its accuracy on validation data
    _,acc = model.evaluate(val_X, val_Y)
    print('The accuracy on the validation data is {}.'.format(acc*100))
    # Save the model
    model.save('model1_10epoch.h5')


if __name__ == '__main__':
    train()
