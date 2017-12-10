import os
import pickle
import h5py
from utils.load_transform_data import dismiss_shorter_tracks,\
    filter_short_files_and_ids, get_target_variable_for_windows_categorical
from utils.load_transform_data import get_target_variables_per_group,\
    create_windows_from_spectrogram
from load_small_dataset import tracks, train_valid_tracks, x_train_ids,\
    validation_valid_tracks, x_val_ids, test_valid_tracks, x_test_ids


def build_spectrograms(nb_windows=10, hop_length=350, length_threshold=27):

    pickled_spectrograms = [f for f in os.listdir('spectrograms')]
    pickle_name = "{}_{}.h5".format(hop_length, nb_windows)

    small_data_set_spectograms = {}
    if pickle_name in pickled_spectrograms:
        print("\n Found spectrograms for hop_length {} and windows {}"
              .format(hop_length, nb_windows))

        datafile = h5py.File("spectrograms/{}".format(pickle_name), 'r')
        small_data_set_spectograms['train'] = (datafile['x_train'][:],
                                               datafile['y_train'][:])

        small_data_set_spectograms['val'] = (datafile['x_val'][:],
                                             datafile['y_val'][:])

        small_data_set_spectograms['test'] = (datafile['x_test'][:],
                                              datafile['y_test'][:])

        return small_data_set_spectograms

    spectrograms, shorter_files =\
        dismiss_shorter_tracks(train_valid_tracks, hop_length,
                               length_threshold)

    filtered_train_ids = filter_short_files_and_ids(x_train_ids,
                                                    shorter_files)

    spectrograms_val, shorter_files_val =\
        dismiss_shorter_tracks(validation_valid_tracks, hop_length,
                               length_threshold)

    filtered_val_ids = filter_short_files_and_ids(x_val_ids,
                                                  shorter_files_val)

    spectrograms_test, shorter_files_test =\
        dismiss_shorter_tracks(test_valid_tracks, hop_length,
                               length_threshold)

    filtered_test_ids = filter_short_files_and_ids(x_test_ids,
                                                   shorter_files_val)

    y_train, y_val, y_test =\
        get_target_variables_per_group(tracks, filtered_train_ids,
                                       filtered_val_ids, filtered_test_ids)

    train_sequences = create_windows_from_spectrogram(spectrograms,
                                                      nb_windows=nb_windows)
    y_train_binary_sequences =\
        get_target_variable_for_windows_categorical(y_train)

    small_data_set_spectograms['train'] = (train_sequences,
                                           y_train_binary_sequences)

    window_size = train_sequences.shape[1]

    val_sequences = create_windows_from_spectrogram(spectrograms_val,
                                                    window_size=window_size,
                                                    nb_windows=nb_windows)
    y_val_binary_sequences = get_target_variable_for_windows_categorical(y_val)

    small_data_set_spectograms['val'] = (val_sequences,
                                         y_val_binary_sequences)

    test_sequences = create_windows_from_spectrogram(spectrograms_test,
                                                     window_size=window_size,
                                                     nb_windows=nb_windows)
    y_test_binary_sequences =\
        get_target_variable_for_windows_categorical(y_test)

    small_data_set_spectograms['test'] = (test_sequences,
                                          y_test_binary_sequences)

    assert train_sequences.shape[1] == val_sequences.shape[1] ==\
        test_sequences.shape[1]

    datafile = (h5py.File("spectrograms/{}_{}.h5".format(hop_length, nb_windows)))
    dataset = (datafile.create_dataset('x_train', train_sequences.shape,
               dtype='f'))
    for index, vector in enumerate(train_sequences):
        dataset[index] = vector

    dataset = (datafile.create_dataset('y_train',
                                       y_train_binary_sequences.shape,
                                       dtype='f'))
    for index, vector in enumerate(y_train_binary_sequences):
        dataset[index] = vector

    dataset = (datafile.create_dataset('x_val', val_sequences.shape,
                                       dtype='f'))
    for index, vector in enumerate(val_sequences):
        dataset[index] = vector

    dataset = (datafile.create_dataset('y_val',
                                       y_val_binary_sequences.shape,
                                       dtype='f'))
    for index, vector in enumerate(y_val_binary_sequences):
        dataset[index] = vector

    dataset = (datafile.create_dataset('x_test', test_sequences.shape,
                                       dtype='f'))
    for index, vector in enumerate(test_sequences):
        dataset[index] = vector

    dataset = (datafile.create_dataset('y_test',
                                       y_test_binary_sequences.shape,
                                       dtype='f'))
    for index, vector in enumerate(y_test_binary_sequences):
        dataset[index] = vector

    datafile.close()

    return small_data_set_spectograms

