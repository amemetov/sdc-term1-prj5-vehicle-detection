import glob
import os

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from skimage import exposure

# import Keras modules
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# prepare data
def read_files_list(base_dir, dirs):
    files = []

    for curr_dir in dirs:
        files.extend(glob.glob(os.path.join(base_dir, curr_dir, '*')))

    return files


def add_conv_layer(model, nb_filter, filter_size, stride, activation, use_bn, dropout_prob,
                   pool_size=0, pool_stride=2, input_dim=None):
    if input_dim is None:
        model.add(Conv2D(nb_filter, filter_size, filter_size, subsample=(stride, stride)))
    else:
        model.add(Conv2D(nb_filter, filter_size, filter_size, subsample=(stride, stride), input_shape=input_dim))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation(activation))
    # model.add(ELU())

    if pool_size > 0:
        model.add(
            MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_stride, pool_stride), border_mode='valid'))

    model.add(Dropout(dropout_prob))


def add_fc_layer(model, nb_hidden_units, activation, use_bn, dropout_prob):
    model.add(Dense(nb_hidden_units))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(ELU())
    model.add(Dropout(dropout_prob))


def build_model(input_dim, conv_activation='relu', fcn_activation='relu', dropout_prob=0.5, use_bn=False):
    model = Sequential()

    # CONV1 -> RELU -> DROPOUT
    add_conv_layer(model, 16, 3, 1, conv_activation, use_bn, dropout_prob, pool_size=0, input_dim=input_dim)
    # CONV2 -> RELU -> MAX_POOL -> DROPOUT
    add_conv_layer(model, 16, 3, 1, conv_activation, use_bn, dropout_prob, pool_size=2)
    # CONV3 -> RELU -> DROPOUT
    add_conv_layer(model, 64, 3, 1, conv_activation, use_bn, dropout_prob, pool_size=0)
    # CONV4 -> RELU -> MAX_POOL -> DROPOUT
    add_conv_layer(model, 64, 3, 1, conv_activation, use_bn, dropout_prob, pool_size=2)

    model.add(Flatten())

    add_fc_layer(model, 256, fcn_activation, use_bn, dropout_prob)
    add_fc_layer(model, 256, fcn_activation, use_bn, dropout_prob)

    model.add(Dense(2))
    model.add(Activation("softmax"))

    return model


def cnn_model(input_dim):
    model = Sequential()

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_dim, activation='relu'))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    # plt.show()
    plt.savefig("./loss-acc-curve.png")
    plt.close()


def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]
    return (accuracy * 100)


def rgb2gray(rgb):
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    # Y' = 0.299 R + 0.587 G + 0.114 B
    result = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return result.reshape((rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))


def preprocess_imgs(imgs, gray=True, hist_eq=True):
    if issubclass(imgs.dtype.type, np.integer):
        # print('Normalizing to the range [0, 1]')
        imgs = imgs / 255.0

    if gray:
        # imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)
        imgs = rgb2gray(imgs)

    if hist_eq:
        imgs = exposure.equalize_hist(imgs)

    return imgs


def predict(model, imgs, gray=True, hist_eq=False):
    imgs = preprocess_imgs(imgs, gray, hist_eq)
    return model.predict(imgs)


def predict_classes(model, imgs, gray=True, hist_eq=False):
    imgs = preprocess_imgs(imgs, gray, hist_eq)
    return model.predict_classes(imgs)


if __name__ == '__main__':
    vehicles_base_dir = './data/vehicles'
    vehicles_dirs = ['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']
    vehicles_files = read_files_list(vehicles_base_dir, vehicles_dirs)
    print('Found {0} vehicle images'.format(len(vehicles_files)))

    not_vehicles_base_dir = './data/non-vehicles'
    not_vehicles_dirs = ['Extras', 'GTI']
    not_vehicles_files = read_files_list(not_vehicles_base_dir, not_vehicles_dirs)
    print('Found {0} not vehicle images'.format(len(not_vehicles_files)))

    # Prepare data
    print('Loading images into memory...')
    X_cars = []
    for f in vehicles_files:
        X_cars.append(mpimg.imread(f))

    X_not_cars = []
    for f in not_vehicles_files:
        X_not_cars.append(mpimg.imread(f))

    # One-hot encoding
    y_cars = np.full((len(X_cars), 2), [1, 0])
    y_not_cars = np.full((len(X_not_cars), 2), [0, 1])

    # Merge cars and not_cars into one dataset
    X_data = np.concatenate((X_cars, X_not_cars))
    y_data = np.concatenate((y_cars, y_not_cars))

    # Preprocessing
    print('Preprocessing dataset ...')
    gray = True  # True
    hist_eq = False  # True
    X_data = preprocess_imgs(X_data, gray=gray, hist_eq=hist_eq)

    print('X_data.shape: {0}'.format(X_data.shape))
    print('y_data.shape: {0}'.format(y_data.shape))

    # Shuffle the dataset
    print('Shuffle dataset ...')
    X_data, y_data = shuffle(X_data, y_data)

    # Split the dataset into train, valid and test sets
    rand_state = np.random.randint(0, 100)
    X_train_net, X_test_net, y_train_net, y_test_net = train_test_split(X_data, y_data, test_size=0.1,
                                                                        random_state=rand_state)

    rand_state = np.random.randint(0, 100)
    X_train_net, X_valid_net, y_train_net, y_valid_net = train_test_split(X_train_net, y_train_net, test_size=0.1,
                                                                          random_state=rand_state)

    print('Size of train samples: {0}'.format(len(y_train_net)))
    print('Size of valid samples: {0}'.format(len(y_valid_net)))
    print('Size of test samples: {0}'.format(len(y_test_net)))

    # Build CNN
    if gray:
        input_dim = (64, 64, 1)
    else:
        input_dim = (64, 64, 3)
    model = cnn_model(input_dim)
    # model = build_model(input_dim, dropout_prob=0.75)

    # Compile model
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 128
    epochs = 10

    checkpoint = ModelCheckpoint('./model.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')

    augment_data = True  # False

    if augment_data:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            shear_range=np.radians(15),
            zoom_range=0.1,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train_net)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(X_train_net, y_train_net, batch_size=batch_size),
                                      samples_per_epoch=X_train_net.shape[0],
                                      nb_epoch=epochs,
                                      validation_data=(X_valid_net, y_valid_net),
                                      callbacks=[checkpoint, early_stopping])
    else:
        print('Not using augmentation')
        history = model.fit(X_train_net, y_train_net, batch_size=batch_size, nb_epoch=epochs,
                            validation_data=(X_valid_net, y_valid_net),
                            callbacks=[checkpoint, early_stopping])

    plot_model_history(history)

    print("Accuracy on test data is: {0:.2f}".format(accuracy(X_test_net, y_test_net, model)))
