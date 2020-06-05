import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers,losses
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
import graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from DeepLearnningAssignment.L4W2A2.kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    X_input = Input(input_shape)
    # X = ZeroPadding2D((3,3))(X_input)
    Z1 = Conv2D(32, (3, 3), strides=(1, 1), name="conv0", padding="SAME")(X_input)
    bn0 = BatchNormalization(name="bn0")(Z1)
    A1 = Activation("relu")(bn0)

    pool1 = MaxPooling2D((2, 2), name="maxpool")(A1)
    # pool1 = GlobalMaxPooling2D()(A1)

    flaten1 = Flatten()(pool1)
    Z2 = Dense(1, activation="sigmoid", name="fc")(flaten1)

    model = Model(inputs=X_input, outputs=Z2, name="happy model")

    ### END CODE HERE ###

    return model

if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    happyModel = HappyModel((64, 64, 3))
    happyModel.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy", metrics=['accuracy'])
    happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)
    ### START CODE HERE ### (1 line)
    preds = happyModel.evaluate(x=X_test, y=Y_test)
    ### END CODE HERE ###
    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    pass