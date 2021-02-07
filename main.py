import keras
import os

import scipy
from sklearn.decomposition import PCA
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
import shutil
from enum import Enum
import sklearn
import random
import numpy as np
import opendatasets as ods
import librosa


# Links to source:
# https://github.com/tushar2025/Speech-Emotion-Detection/blob/master/Neural%20Net%20Training.ipynb
# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
# https://www.intechopen.com/books/social-media-and-machine-learning/automatic-speech-emotion-recognition-using-machine-learning
# https://python-speech-features.readthedocs.io/en/latest/
# https://medium.com/@tushar.gupta_47854/speech-emotion-detection-74337966cf2


class RevdessDataset:
    dataset_name = 'ravdess-emotional-speech-audio'
    dataset_url = os.path.join('https://www.kaggle.com/uwrfkaggler/', dataset_name)

    class Identfier(Enum):
        EMOTION = 2
        INTENSITY = 3
        STATEMENT = 4
        REPETITION = 5
        ACTOR = 6

    def __init__(self, data_dir):
        self.dataset_path = os.path.join(data_dir, self.dataset_name)
        if not os.path.exists(self.dataset_path):
            ods.download(self.dataset_url, data_dir)
            shutil.rmtree(os.path.join(self.dataset_path, shutil.rmtree("")))
        else:
            print("Data set already downloaded.")
        self.files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                self.files.append(os.path.join(root, file))

    def query(self, emotion=None, intensity=None, statement=None, repetition=None, actor=None):
        query_result = self.files
        if emotion is not None:
            query_result = self.__query_param__(2, emotion, query_result)
        if intensity is not None:
            query_result = self.__query_param__(3, intensity, query_result)
        if statement is not None:
            query_result = self.__query_param__(4, statement, query_result)
        if repetition is not None:
            query_result = self.__query_param__(5, repetition, query_result)
        if actor is not None:
            query_result = self.__query_param__(6, actor, query_result)
        return query_result

    def get_ratios(self, query_result, identifier):
        summation = len(query_result)
        ratio_dict = {}
        for file in query_result:
            key = os.path.basename(file).split('-')[identifier.value]
            if key in ratio_dict.keys():
                ratio_dict[key] += 1
            else:
                ratio_dict[key] = 1
        return ratio_dict

    def __query_param__(self, type_index, values, query):
        return list(filter(lambda file: os.path.basename(file).split('-')[type_index] in values, query))


def add_noise_from_file(x_data, y_data, noise_file, percent_of_data, max_pitch_vary, max_db_vary):



def get_features(data_x):
    new_data_x = []
    for data in tqdm(data_x):
        new_data_x.append(compute_feature_vector(data, 12))
    return new_data_x


def compute_feature_vector(data, mfcc_coefficent_n):
    feature_vector = np.array([])
    signal = np.array(data[0])
    sr = data[1]

    # MFCC
    M = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=mfcc_coefficent_n)
    for i in range(M.shape[0]):
        coefficent = M[i, :]
        feature_vector = np.hstack((feature_vector, np.mean(coefficent)))
        feature_vector = np.hstack((feature_vector, np.std(coefficent)))
        feature_vector = np.hstack((feature_vector, scipy.stats.kurtosis(coefficent)))
        feature_vector = np.hstack((feature_vector, scipy.stats.skew(coefficent)))

    # Chroma
    stft = np.abs(librosa.stft(signal))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    feature_vector = np.hstack((feature_vector, chroma))

    # Mel Scale
    mel = np.mean(librosa.feature.melspectrogram(signal, sr=sr).T, axis=0)
    feature_vector = np.hstack((feature_vector, mel))

    return feature_vector


def extract_data_from_file(files, label):
    x_data = []
    y_data = []
    for file in tqdm(files):
        signal, sr = librosa.load(file, mono=True)
        x_data.append([signal, sr])
        y_data.append(label)
    return x_data, y_data


def duplicate_data(x_data, y_data, length=0.5, times=4):
    x_new_data = []
    y_new_data = []
    for i in tqdm(range(len(x_data))):
        signal, sr = x_data[i]
        signal_length = len(signal)
        label = y_data[i]
        new_length = int(np.floor(signal_length * length))
        start_point = 0
        interval = int(np.floor((signal_length * (1 - length)) / times - 1))
        for time in range(times):
            x_new_data.append([signal[start_point: start_point + new_length], sr])
            y_new_data.append(label)
            start_point += interval

    return x_new_data, y_new_data


def vary_pitch(x_data, y_data, max_pitch_diff=3, new_pitch_per_data=3):
    x_new_data = []
    y_new_data = []
    for i in tqdm(range(len(x_data))):
        signal, sr = x_data[i]
        for pitch in range(new_pitch_per_data):
            new_signal = librosa.effects.pitch_shift(signal, sr, random.uniform(-max_pitch_diff, max_pitch_diff))
            x_new_data.append([new_signal, sr])
            y_new_data.append(y_data[i])

    return x_new_data, y_new_data


def shuffle_data(x_data, y_data, times=20):
    for i in tqdm(range(20)):
        x_data, y_data = sklearn.utils.shuffle(x_complete_data, y_complete_data)

    return x_data, y_data


x_complete_data = []
y_complete_data = []
angry_label = 1
not_angry_label = 0
x_train = []
x_test = []
y_train = []
y_test = []

# create workspace structure
resource_path = os.path.join(os.getcwd(), 'resources')
data_files_path = os.path.join(resource_path, 'data_files')
if not os.path.isdir(resource_path):
    os.mkdir(data_files_path)
if not os.path.isdir(resource_path):
    os.mkdir(data_files_path)

if len(os.listdir(data_files_path)) == 0:
    print("Get Ravdess dataset")
    data_set = RevdessDataset(resource_path)

    # debug
    print(len(compute_feature_vector(librosa.load(data_set.files[0], mono=True), 12)))

    print("Extract files with label: 1")
    x, y = extract_data_from_file(data_set.query(emotion=['05', '08'], intensity=['02']), 1)
    x_complete_data = x_complete_data + x
    y_complete_data = y_complete_data + y

    print("Extract files with label: 0")
    x, y = random.sample(
        extract_data_from_file(data_set.query(emotion=['01', '02', '03', '04', '05', '06'], intensity=['01', '02']), 0),
        len(x_complete_data))
    x_complete_data = x_complete_data + x
    y_complete_data = y_complete_data + y

    print("Duplicate data data set:")
    x_complete_data, y_complete_data = duplicate_data(x_complete_data, y_complete_data)

    print("Add some pitched examples:")
    x, y = vary_pitch(x_complete_data, y_complete_data)
    x_complete_data = x_complete_data + x
    y_complete_data = y_complete_data + y

    print("Add clean noisy data:")
    x, y = add_noise_from_file(x_complete_data, y_complete_data, os.path.join(resource_path, "clean_noise.wav"), 1.0, 0.8, 10)
    x_complete_data = x_complete_data + x
    y_complete_data = y_complete_data + y

    print("Add keyboard noise:")
    x, y = add_noise_from_file(x_complete_data, y_complete_data, os.path.join(resource_path, "keyboard_sound.wav"), 0.1, 0.5, 10)
    x_complete_data = x_complete_data + x
    y_complete_data = y_complete_data + y

    print("Add mouse click noise:")
    x, y = add_noise_from_file(x_complete_data, y_complete_data, os.path.join(resource_path, "mouse_click_sound.wav"), 0.05, 0.5, 10)
    x_complete_data = x_complete_data + x
    y_complete_data = y_complete_data + y


    print("Add error noise:")

    print("Compute features from signal:")
    x_complete_data = get_features(x_complete_data)

    print("Shuffle data:")
    x_complete_data, y_complete_data = shuffle_data(x_complete_data, y_complete_data, 200)

    print("Separate test set from data(80/20)")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_complete_data, y_complete_data,
                                                                                test_size=0.2)

    print("Save as csv")
    os.chdir(data_files_path)
    np.save('x_train', np.asarray(x_train), True)
    np.save('y_train', np.asarray(y_train), True)
    np.save('x_test', np.asarray(x_test), True)
    np.save('y_test', np.asarray(y_test), True)
    os.chdir("../..")
else:
    os.chdir(data_files_path)
    x_train = np.load('x_train.npy', mmap_mode=None, allow_pickle=True)
    y_train = np.load('y_train.npy', mmap_mode=None, allow_pickle=True)
    x_test = np.load('x_test.npy', mmap_mode=None, allow_pickle=True)
    y_test = np.load('y_test.npy', mmap_mode=None, allow_pickle=True)
    os.chdir("../..")


def evaluate_model(trainX, trainy, testX, testy):
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same', input_shape=(trainX.shape[1], 1)))
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
    model.fit(trainX, trainy, batch_size=16, epochs=30)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy)
    # save model
    return accuracy


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# pca select main features
pca = PCA(.95)

train_pca = pca.fit_transform(x_train)
test_pca = pca.transform(x_test)

# expand dims
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

repeats = 10
scores = list()
for r in range(repeats):
    score = evaluate_model(x_train, y_train, x_test, y_test)
    score = score * 100.0
    print('>#%d: %.3f' % (r + 1, score))
    scores.append(score)
# summarize results
summarize_results(scores)
