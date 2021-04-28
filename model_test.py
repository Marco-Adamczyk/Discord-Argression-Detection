import os
import numpy as np
import librosa
import random

from sklearn.decomposition import PCA
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import data_processing


def test_model(model_path, test_sound_file_path, label, repetition, signal_length, take_components=None, training_path=None):
    if not os.path.isfile(model_path):
        raise Exception("No model file found")
    if not os.path.isfile(test_sound_file_path):
        raise Exception("No model file found")
    loaded_model = load_model(model_path)

    # select random samples and its features from the test file with the signal length matching the trained length
    x_test = []
    signal, sr = librosa.load(test_sound_file_path)
    for i in range(repetition):
        start_index = random.randint(0, len(signal) - signal_length - 1)
        new_signal = signal[start_index:start_index + signal_length]
        x_test.append([new_signal, sr])
    x_test = data_processing.get_features(x_test)

    # transform the test set with pca according to the size of the loaded model input dim. To ensure the
    # same dim is taken the original train set has to be fitted.
    if take_components is not None:
        x_train = np.load(training_path, mmap_mode=None, allow_pickle=True)
        pca = PCA(n_components=take_components)
        pca.fit(x_train)
        x_test = pca.transform(x_test)

    # expand dims
    x_test = np.expand_dims(x_test, axis=2)

    # create the categorical vectors with expected label
    y_test = to_categorical(np.full((len(x_test)), label), num_classes=2)

    # evaluate
    _, accuracy = loaded_model.evaluate(x_test, y_test, batch_size=16)


signal, _ = librosa.load(
    os.path.join(data_processing.resource_path, 'ravdess-emotional-speech-audio', 'Actor_01', '03-01-01-01-01-01-01.wav'))
signal_length = int(len(signal) / 2)

training_path = os.path.join(os.getcwd(), data_processing.data_files_path, 'x_train.npy')

test_sound_file_path = os.path.join(data_processing.resource_path, 'calm_test_voice.wav')
test_sound_file_path4 = os.path.join(data_processing.resource_path, 'calm_test_voice_2.wav')
test_sound_file_path2 = os.path.join(data_processing.resource_path, 'agressive_test_voice.wav')
test_sound_file_path3 = os.path.join(data_processing.resource_path, 'agressive_test_voice_2.wav')

# First model without pca and a rather imbalanced dataset 30 epochs
model_path = os.path.join(data_processing.net_files_path, 'aggression_detect_model.h5')
test_model(model_path, test_sound_file_path, 0, 30, signal_length)
test_model(model_path, test_sound_file_path2, 1, 30, signal_length)


#test_model(model_path, test_sound_file_path3, 1, 15, signal_length)
#test_model(model_path, test_sound_file_path4, 0, 15, signal_length)

# Second model pca 80 and epochs 40
#model_path = os.path.join(data_processing.net_files_path, 'aggression_detect_model2_pca_80.h5')
#test_model(model_path, test_sound_file_path, 0, 30, signal_length, 80, training_path)
#test_model(model_path, test_sound_file_path2, 1, 30, signal_length, 80, training_path)

# Second model pca 80 and epochs 30
#model_path = os.path.join(data_processing.net_files_path, 'aggression_detect_model3_pca_40_epoch_30.h5')
#test_model(model_path, test_sound_file_path, 0, 30, signal_length, 40, training_path)
#test_model(model_path, test_sound_file_path2, 1, 30, signal_length, 40, training_path)

# Second model pca 80 and epochs 30
#model_path = os.path.join(data_processing.net_files_path, 'aggression_detect_model4_epoch_33.h5')
#test_model(model_path, test_sound_file_path, 0, 30, signal_length)
#test_model(model_path, test_sound_file_path2, 1, 30, signal_length)