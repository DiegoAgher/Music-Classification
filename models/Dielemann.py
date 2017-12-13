import keras.backend as K_backend
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten,\
    GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda
from keras.layers import merge


def l2_norm(x):
    x **= 2
    x = K_backend.sum(x, axis=1)
    x = K_backend.sqrt(x)
    return x


class DielmannArq(object):
    def __init__(self, nb_filters_1=10, nb_filters_2=20, lenght_filters=4,
                 pool_size_1=4, pool_size_2=2, dropout_prob_1=0.5,
                 dropout_prob_2=0.5, frames=2580, frequency=128,
                 dense_size=100, nb_classes=8):
        self.nb_filters_1 = nb_filters_1
        self.nb_filters_2 = nb_filters_2
        self.lenght_filters = lenght_filters
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.dropout_prob_1 = dropout_prob_1
        self.dropout_prob_2 = dropout_prob_2
        self.frames = frames
        self.frequency = frequency
        self.dense_size = dense_size
        self.nb_classes = nb_classes

    def build_convolutional_model(self):
        input_placeholder = Input(shape=(self.frames, self.frequency))

        conv_1 = Conv1D(self.nb_filters_1, self.lenght_filters,
                        activation='relu', border_mode='same')(
            input_placeholder)
        pool_1 = MaxPooling1D(self.pool_size_1)(conv_1)

        conv_2 = Conv1D(self.nb_filters_1, self.lenght_filters,
                        border_mode='same', activation='relu')(pool_1)
        pool_2 = MaxPooling1D(self.pool_size_1)(conv_2)

        conv_3 = Conv1D(self.nb_filters_2, self.lenght_filters,
                        border_mode='same', activation='relu')(pool_2)
        pool_3 = MaxPooling1D(self.pool_size_2)(conv_3)

        global_mean = GlobalAveragePooling1D()(pool_3)
        global_max = GlobalMaxPooling1D()(pool_3)

        concat = merge([global_mean, global_max], mode='concat',
                       concat_axis=-1)

        hidden = Dense(self.dense_size, activation='relu')(concat)
        drop_1 = Dropout(self.dropout_prob_1)(hidden)
        hidden_2 = Dense(self.dense_size, activation='relu')(drop_1)
        drop_2 = Dropout(self.dropout_prob_1)(hidden_2)

        output = Dense(self.nb_classes, activation='softmax')(drop_2)

        model = Model(input=input_placeholder, output=output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        return

    def fit(self, x, y, batch_size, nb_epoch, validation_split):
        self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,
                       validation_split=validation_split)
        return
