from keras.models import Sequential

from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Input, Lambda, SpatialDropout2D
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import *

BATCH_SIZE = 64
EPOCHS = 50
LOSS_RATE = 0.001
LOSS_FUNCTION = 'mse'
SAME_BORDER_MODE = 'same'
VALID_BORDER_MODE = 'valid'
SUB_SAMPLE = (2,2)
ACTIVATION = 'elu'


"""
        Returns an NVIDIA Convolutional Neural Network given an image input_shape
"""
def model_nvidia(input_shape):
    
    
    def resize_images(img):
        import tensorflow as tf
        return tf.image.resize_images(img, (66, 200))
    
    
    
    model = Sequential()
    model.add(Lambda(resize_images, input_shape=input_shape))
    model.add(Lambda(lambda x: x/255.-0.5))
    model.add(Convolution2D(24, 5, 5, border_mode=SAME_BORDER_MODE, subsample=SUB_SAMPLE, activation=ACTIVATION))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode=SAME_BORDER_MODE, subsample=SUB_SAMPLE, activation=ACTIVATION))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode=VALID_BORDER_MODE, subsample=SUB_SAMPLE, activation=ACTIVATION))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode=VALID_BORDER_MODE, activation=ACTIVATION))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode=VALID_BORDER_MODE, activation=ACTIVATION))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation=ACTIVATION))
    model.add(Dense(50, activation=ACTIVATION))
    model.add(Dense(10, activation=ACTIVATION))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(lr=LOSS_RATE), loss=LOSS_FUNCTION)
    return model

## Load driving data from csv file "../data/driving_log.csv" into pandas dataframe
csv_df = pd.read_csv("data/driving_log.csv", index_col=False)

# set the data frame columns
csv_df.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']

## Shuffle
csv_df = csv_df.sample(n=len(csv_df))

## Training and Validation Data
training_count = int(0.8 * len(csv_df))
training_data, validation_data = csv_df[:training_count].reset_index(), csv_df[training_count:].reset_index()


def drop_small_steering_values(data):
    """ Reduce training data where steering changes are small
        i.e. if the car remains straight, then subsequent images
        are relatively similar. As a result, these can be omitted.
        """
    rows = []
    for steering_angle in data[abs(data['steer'])<.05].index.tolist():
        if np.random.randint(10) < 8:
            rows.append(steering_angle)
    data = data.drop(data.index[rows])
    print("Reduced driving data by %s rows with relatively low steering"%(len(rows)))
    return data

def samples_per_epoch(training_data, batch_size):
    return int(len(training_data) / batch_size) * batch_size
## Reducing low steering angle data to remove bias
training_data = drop_small_steering_values(training_data)

img = process_image_from_path(training_data['center'][5].strip())

## Create NVIDIA Convolutional Neural Network Model
model = model_nvidia(img.shape)
samples_per_epoch_ = samples_per_epoch(training_data, BATCH_SIZE)
nb_val_samples = len(validation_data)
values = model.fit_generator(trainer(training_data, BATCH_SIZE), samples_per_epoch=samples_per_epoch_, nb_epoch=EPOCHS, validation_data=get_validation_data(validation_data), nb_val_samples=len(validation_data))

model.save('model.h5')
print(model.summary())
