import numpy as np
from utils.load_transform_data import average_window_array,\
    to_categorical_inverse, enc
from models.Dielmann import DielmannArq
from build_spectrograms import build_spectrograms


windows_number = 10
songs_size = 30

small_dataset_spectrograms = build_spectrograms(windows_number, 350)

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
                     nb_epoch=1,
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

classifier.model.save('my_classifier.h5')



