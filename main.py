import numpy as np
import helpers
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, \
    Dropout, Flatten

batch_size = 32
nb_classes = 2
nb_epoch = 200


def get_model(X_train):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    return model


def compile_model(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


def check_classificator(X_train, Y_train):
    X_train = np.array(X_train).astype('float32')
    X_train /= 255
    print(X_train.shape)
    model = get_model(X_train)
    compile_model(model)


    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        shuffle=True)
    print(history)


if __name__ == '__main__':
    X_train, Y_train = helpers.get_data()
    check_classificator(X_train, Y_train)
