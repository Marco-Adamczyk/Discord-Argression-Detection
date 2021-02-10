import os

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.python.estimator import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.python.keras.utils.np_utils import to_categorical
import data_processing


def learn_model(x_train, y_train, x_test, y_test, take_components, save_path=None):
    # pca select main features
    pca = PCA(n_components=take_components)
    print("Compute pca relevant features with " + str(take_components) + " percent of variance")
    previous_dims = len(x_train[0])
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    print(str(len(x_train[0])) + " dims are used from initially " + str(previous_dims))

    # expand dims
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    # change label to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # build model
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same', input_shape=(x_train.shape[1], 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0),
                  metrics=['acc'])

    # fit network
    model.fit(x_train, y_train, batch_size=16, epochs=40)

    # evaluate model
    _, accuracy = model.evaluate(x_test, y_test)

    # save model
    if save_path is not None:
        model.save(save_path)

    return accuracy


# learn
x_train, y_train, x_test, y_test = data_processing.get_processed_data(data_processing.resource_path, data_processing.data_files_path)
net_file = os.path.join(data_processing.save_path, 'aggression_detect_model2_pca_80.h5')
print(learn_model(x_train, y_train, x_test, y_test, 80, net_file))