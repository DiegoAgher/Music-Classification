from fma import utils
from utils.load_transform_data import VALID_GENRES, get_tracks_locations,\
    get_filename_and_ids


music_files_locations = get_tracks_locations()

tracks = utils.load('fma/tracks.csv')
small = tracks['set', 'subset'] <= 'small'
train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'
selected_genres = tracks['track', 'genre_top'].isin(VALID_GENRES)

x_train_ids = tracks.loc[small & train & selected_genres].index.values
x_val_ids = tracks.loc[small & val & selected_genres].index.values
x_test_ids = tracks.loc[small & test & selected_genres].index.values

train_valid_tracks = get_filename_and_ids(x_train_ids, music_files_locations)
validation_valid_tracks = get_filename_and_ids(x_val_ids,
                                               music_files_locations)

test_valid_tracks = get_filename_and_ids(x_test_ids, music_files_locations)
