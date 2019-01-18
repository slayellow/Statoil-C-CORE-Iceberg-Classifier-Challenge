#Import Keras.
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(csv):
    temp = pd.read_json(csv)
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in temp["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in temp["band_2"]])
    X_data = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],
                              ((X_band_1 + X_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)
    if 'train' in csv:
        target = temp['is_iceberg']
    else:
        target = temp['id']
    print('X_band_1 : {}'.format(X_band_1.shape))
    print('X_band_2 : {}'.format(X_band_2.shape))
    print('X_train : {}'.format(X_data.shape))

    return X_data, target


# define our model
def getModel():
    # Building the model
    gmodel = Sequential()
    # Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    # Conv Layer 2
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    # Conv Layer 3
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    # Conv Layer 4
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    # Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    # Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    # Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    # Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                   optimizer=mypotim,
                   metrics=['accuracy'])
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def evaluate_prediction(prediction):
    prediction = prediction.reshape((prediction.shape[0]))
    result = []
    for i, predict in enumerate(prediction):
        if predict >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


def main():
    X_train, Y = get_data("E:/Data/Statoil C-CORE Iceberg Classifier Challenge/train.json")
    X_test, testID = get_data("E:/Data/Statoil C-CORE Iceberg Classifier Challenge/test.json")

    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, Y, random_state=1, train_size=0.7)

    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)
    gmodel = getModel()
    while True:
        try:

            Input = int(input("You can select number. Training[1] / Test&Save[2] / Exit[3]"))
            if Input == 1:
                gmodel.fit(X_train_cv, y_train_cv,
                           batch_size=12,
                           epochs=50,
                           verbose=1,
                           validation_data=(X_valid, y_valid),
                           callbacks=callbacks)

                gmodel.load_weights(file_path)
                score = gmodel.evaluate(X_valid, y_valid, verbose=1)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
            elif Input == 2:
                predicted_test = gmodel.predict_proba(X_test)
                submission = pd.DataFrame()
                submission['id'] = testID
                submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
                submission.to_csv('sub.csv', index=False)
                print('Test Complete')
            elif Input == 3:
                break
            else:
                raise Exception('\n \t\t[1]Training,\t[2]Test&Save,\t[3]Exit\t')

        except NameError as err:
            print("1 OR 2 만 입력해주세요 ")

        except ValueError as err:
            print("only number")

        except KeyboardInterrupt:
            print("retry")

if __name__ == "__main__":
    main()


