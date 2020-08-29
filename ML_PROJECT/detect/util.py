from __future__ import division, absolute_import, print_function

import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import scale
import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.128, 'bim-b': 0.265},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.132, 'bim-a': 0.015, 'bim-b': 0.122}
}
# Set random seed
np.random.seed(0)


def get_data():
    dataset='mnist'
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to (n_samples, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test


def get_model():
    """
    return: The model; a Keras 'Sequential' instance.
    """
    dataset='mnist'
    # MNIST model
    layers = [
        Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),            
        Dense(10),
        Activation('softmax')
    ]
    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model


def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    output_dim = model.layers[-1].output.shape[-1].value
    get_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output]
    )

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)


def normalize(uncerts_all):
    total = scale(uncerts_all)

    return total
