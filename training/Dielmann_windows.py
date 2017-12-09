import sys
import numpy as np
from utils.load_transform_data import average_window_array,\
    to_categorical_inverse, enc
from models.Dielmann import DielmannArq
from build_spectrograms import build_spectrograms


def assign_params(parameters):
    hop_length = 350
    windows_number = 10
    model_name = 'my_classifier.h5'
    number_params = len(parameters)
    if number_params >= 2:
         hop_length = int(sys.argv[1])
    if number_params >= 3:
        windows_number = int(sys.argv[2])

    if number_params >= 4:
        print("went in, {} is the param".format(sys.argv[3]))
        model_name = str(sys.argv[3])

    if windows_number > 10:
        windows_number = 10
        print("Tracks are 30 seconds long, "
              "can't create more than 10 windows per track, using"
              " windows_number=10")

    return hop_length, windows_number, model_name

training_params = sys.argv
hop_length, windows_number, model_name = assign_params(training_params)


print("\n Using hop_length {} for spectrogram calculation; "
      "Number of windows per track {};"
      "Model name {}".format(hop_length, windows_number, model_name))

small_dataset_spectrograms = build_spectrograms(windows_number, hop_length)

train_sequences, y_train_binary_sequences = small_dataset_spectrograms['train']
val_sequences, y_val_binary_sequences = small_dataset_spectrograms['val']
test_sequences, y_test_binary_sequences = small_dataset_spectrograms['test']

frame_size = train_sequences.shape[1]


classifier = DielmannArq(frames=frame_size, nb_filters_1=128,
                         nb_filters_2=128*2, dense_size=1024,
                         dropout_prob_1=0.2, nb_classes=4)

classifier.build_convolutional_model()
classifier.model.fit(train_sequences, y_train_binary_sequences,
                     batch_size=32,
                     nb_epoch=10,
                     validation_data=(val_sequences, y_val_binary_sequences))

loss, accuracy = classifier.model.evaluate(test_sequences,
                                           y_test_binary_sequences)


print("Loss: {} ; Acc: {} on windowed test dataset ".format(loss, accuracy))

window_probabilities = classifier.model.predict(test_sequences)

averaged_probabilities = average_window_array(window_probabilities)
predictions = np.argmax(averaged_probabilities, axis=1)

y_test = to_categorical_inverse(y_test_binary_sequences)
number_songs = int(y_test.shape[0] / windows_number)

indices = [windows_number * x for x in range(number_songs)]
y_test = y_test[indices]

accuracy = sum(predictions == y_test) / len(y_test)

print("Accuracy for full songs is {}".format(accuracy))

classifier.model.save(model_name)

