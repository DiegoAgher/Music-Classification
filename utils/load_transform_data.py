import os
import librosa
import numpy as np
import sklearn as skl
from keras.utils.np_utils import to_categorical

VALID_GENRES = ['Electronic', 'Instrumental', 'Rock', 'Hip-Hop']
AUDIO_DIR = 'fma/fma_small'


def get_tracks_locations():
    music_files_locations = []
    for dir_name in os.listdir(AUDIO_DIR):
        if dir_name not in ['README.txt', 'checksums']:
            file_names = os.listdir(AUDIO_DIR + '/' + dir_name)
            current_files = [AUDIO_DIR + '/' + dir_name + '/' + filename
                             for filename in file_names]
            music_files_locations += current_files

    print("Read {0} files".format(len(music_files_locations)))
    return music_files_locations


def get_filename_and_ids(track_ids_group, music_files_locations):
    return [(filename, int(filename.split('/')[-1][:6]))
            for filename in music_files_locations
            if int(filename.split('/')[-1][:6]) in track_ids_group]


def dismiss_shorter_tracks(valid_track_and_ids_list, hop_length,
                           length_threshold=27):
    shorter_files = []
    spectograms = {}
    for index, (filename, track_id) in enumerate(valid_track_and_ids_list):
        if index % 100 == 0:
            print("reading file number: ", index)
        x, sr = librosa.load(filename, sr=None, mono=True)
        if x.shape[-1] / sr >= length_threshold:
            mel = librosa.feature.melspectrogram(y=x, sr=sr,
                                                 hop_length=hop_length)
            log_mel = np.transpose(librosa.logamplitude(mel))
            spectograms[track_id] = log_mel
        else:
            print("here's a short one, ", filename)
            shorter_files.append((filename, track_id))
    return spectograms, shorter_files


def filter_short_files_and_ids(track_ids_group, shorter_files):
    shorter_files_ids = [x[1] for x in shorter_files]
    return np.array([x for x in track_ids_group if x not in shorter_files_ids])


enc = skl.preprocessing.LabelEncoder()


def get_target_variables_per_group(tracks_df, x_train_ids, x_val_ids,
                                   x_test_ids):
    y_train = get_target_variable_group(tracks_df, x_train_ids)
    y_train = enc.fit_transform(y_train)
    y_val = get_target_variable_group(tracks_df, x_val_ids)
    y_val = enc.transform(y_val)
    y_test = get_target_variable_group(tracks_df, x_test_ids)
    y_test = enc.transform(y_test)
    return y_train, y_val, y_test


def get_target_variable_group(tracks_df, track_ids_group):
    return tracks_df.loc[track_ids_group, ('track', 'genre_top')]


def create_windows_from_spectrogram(spectrograms_dict, window_size=None,
                                    nb_windows=10, frequency_size=128):
    if window_size is None:
        window_size = int(
            np.floor(next(iter(spectrograms_dict.values())).shape[0] /
                     nb_windows))

    window_sequences = np.zeros((len(spectrograms_dict) * nb_windows,
                                 window_size, frequency_size))

    for idx, (track_id, x) in enumerate(spectrograms_dict.items()):
        for window_nb in range(0, nb_windows):
            spectogram_window = x[window_nb * window_size: (window_nb + 1) *
                                  window_size, :]
            if spectogram_window.shape[0] == window_size:
                window_sequences[idx * nb_windows + window_nb, :, :] =\
                    spectogram_window

            elif spectogram_window.shape[0] < window_size - 1:
                print("spectrogram shorter than expected, id: ", track_id)
                print("appending window of shape: ", spectogram_window.shape)
                window_sequences[idx * nb_windows + window_nb,
                                 :spectogram_window.shape[0], :] =\
                    spectogram_window

    return window_sequences


def get_target_variable_for_windows_categorical(target_variable,
                                                nb_windows=10):
    target_variable_for_windows = []
    for value in target_variable:
        target_variable_for_windows += ([value] * nb_windows)

    return to_categorical(target_variable_for_windows)
