from models.Dielmann import DielmannArq
from build_spectrograms import build_spectrograms

small_dataset_spectrograms = build_spectrograms()

train_sequences, y_train_binary_sequences = small_dataset_spectrograms['train']
val_sequences, y_val_binary_sequences = small_dataset_spectrograms['val']
test_sequences, y_test_binary_sequences = small_dataset_spectrograms['test']


classifier = DielmannArq(frames=377, nb_filters_1=128, nb_filters_2=128*2,
                         dense_size=1024, dropout_prob_1=0.2, nb_classes=4)

classifier.build_convolutional_model()
classifier.model.fit(train_sequences, y_train_binary_sequences,
                     batch_size=32,
                     nb_epoch=10,
                     validation_data=(val_sequences, y_val_binary_sequences))

loss, accuracy = classifier.model.evaluate(test_sequences,
                                           y_test_binary_sequences)

classifier.model.save('my_classifier.h5')



