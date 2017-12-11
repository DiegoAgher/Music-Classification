import numpy as np
from utils.load_transform_data import average_window_array,\
     to_categorical_inverse


def get_full_song_predictions(window_probabilities, windows_y,
                              windows_number=10):
    averaged_probabilities = average_window_array(window_probabilities)
    predictions = np.argmax(averaged_probabilities, axis=1) 
    y = to_categorical_inverse(windows_y)                                                                                     
    number_songs = int(y.shape[0] / windows_number) 
    indices = [windows_number * x for x in range(number_songs)] 
    y = y[indices]                                                                                            
    accuracy = sum(predictions == y) / len(y)
    print("\nAccuracy for full songs is {}".format(accuracy))
    return y, predictions, accuracy
