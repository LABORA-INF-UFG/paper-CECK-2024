from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import argparse
import sys
import efficientnet.keras as efn

# from efficientnet.keras import EfficientNetB0
from efficientnet.keras import EfficientNetB7
from efficientnet.keras import center_crop_and_resize, preprocess_input

width = 450
height = 450
channel = 3

def main(args):
    # dir_images_train = args.dir_images_train
    # dir_images_test = args.dir_images_test
    # (train_images, train_labels), (test_images, test_labels) = load_data(dir_images_train, dir_images_test)
    x_train, x_test, y_train, y_test = load_samples(args.dir_images)
    train_model(x_train, y_train, x_test, y_test)

def load_samples(base_directory_dataset):
    images = np.array([]).reshape(0, height, width, channel)
    directory_dataset_raw = os.path.abspath(os.path.join(base_directory_dataset,'originais_450'))
    filelist_raw = glob.glob(directory_dataset_raw + '/*.jpg')
    labels_0 = np.zeros(len(filelist_raw))
    images_raw = np.array([np.array(Image.open(fname)) for fname in filelist_raw])
    directory_dataset_disturb = os.path.abspath(os.path.join(base_directory_dataset, 'corrompidas_450'))
    filelist_disturb = glob.glob(directory_dataset_disturb + '/*.jpg')
    labels_1 = np.ones(len(filelist_disturb))
    images_disturb = np.array([np.array(Image.open(fname)) for fname in filelist_disturb])
    images = np.append(images, images_raw, axis=0)
    images = np.append(images, images_disturb, axis=0)
    labels = np.append(labels_0, labels_1, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, random_state=42)
    return X_train, X_test, y_train, y_test


def load_data(dir_images_train, dir_images_test):
    X_train, y_train = load_samples(dir_images_train)
    X_test, y_test = load_samples(dir_images_test)
    return (X_train, y_train), (X_test, y_test)




# (train_images, train_labels), (test_images, test_labels) = load_data()
def train_model(train_images, train_labels, test_images, test_labels):
    # train_images = train_images.reshape((train_images.shape[0], height, width, channel))
    # test_images = test_images.reshape((test_images.shape[0], height, width,channel))
    train_images, test_images = train_images / 255.0, test_images / 255.0
    base_model = EfficientNetB7(weights = None, input_shape=(height, width, channel), include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    n_classes = 2
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])

    # train

    # model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_type = 'B7'
    model_name = 'char_recog_ceia_efficientnet_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    # callbacks = [checkpoint, lr_reducer, lr_scheduler]
    callbacks = [checkpoint]
    model.fit(train_images, train_labels, batch_size=2, epochs=100, validation_data=(test_images, test_labels),
              shuffle=True, callbacks=callbacks)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc: %s' % str(test_acc))
    print('test_loss: %s' % str(test_loss))
    model.save("uav_imgage_disturc_class_efficientnet_%s.h5" % model_type)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
