import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

Input_shape = (28, 28, 1)
Num_classes = 10

def get_dataset():
    global Num_classes

    # скачиваем данные и разделяем на надор для обучения и тестов
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    validation_size = int(len(x_train)/len(x_test))
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # преобразование векторных классов в бинарные матрицы
    y_train = keras.utils.to_categorical(y_train, Num_classes)
    y_test = keras.utils.to_categorical(y_test, Num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_validation = x_train[::validation_size]
    y_validation = y_train[::validation_size]

    return (x_train, y_train), (x_test, y_test), (x_validation, y_validation)



def convolutional_neural_network_model():
    global Input_shape
    global Num_classes

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=Input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    start_time = time.time()
    (x_train, y_train), (x_test, y_test), (x_validation, y_validation) = get_dataset()
    model = convolutional_neural_network_model()
    hist = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_validation, y_validation))
    print("Модель успешно обучена")

    model.save('mnist.h5')
    print("Модель сохранена как mnist.h5")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Потери на тесте:', score[0])
    print('Точность на тесте:', score[1])
    print("--- %s seconds ---" % (time.time() - start_time))
