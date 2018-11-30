# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:50:40 2018

@author: Gxy
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
from keras.models import Model
from keras import regularizers
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

def load_mnist_dataset(keep_dim=False):
    (x_train, _), (x_test, _) = mnist.load_data()
    #print(type(mnist)):class 'module'
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if keep_dim:
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    else:
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test

def simplest_possible_autoencoder(sparsity=False):
    encoding_dim = 32
    input_img = Input(shape=(784,))
    if sparsity:
        encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(0.01))(input_img)
    else:
        encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    return encoder, decoder, autoencoder

def sparsity_constraint_autoencoder():
    return simplest_possible_autoencoder(True)

def deep_autoencoder():
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(32,))
    decoder_layers = autoencoder.layers[5:]
    y = autoencoder.layers[4](encoded_input)
    for each in decoder_layers:
        y = each(y)
    decoder = Model(encoded_input, y)
    return encoder, decoder, autoencoder

def convolutional_autoencoder():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(4, 4, 8))
    decoder_layers = autoencoder.layers[8:]
    y = autoencoder.layers[7](encoded_input)
    for each in decoder_layers:
        y = each(y)
    decoder = Model(encoded_input, y)
    return encoder, decoder, autoencoder

def denosing_autoencoder():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(7, 7, 32))
    decoder_layers = autoencoder.layers[6:]
    y = autoencoder.layers[5](encoded_input)
    for each in decoder_layers:
        y = each(y)
    decoder = Model(encoded_input, y)
    return encoder, decoder, autoencoder

def train_autoencoder(autoencoder_list, x_train, x_test, y_train=None, y_test=None, optimizer='adam', loss='binary_crossentropy', epochs=50, batch_size=256):
    autoencoder_list[-1].compile(optimizer=optimizer, loss=loss)
    if y_train:
        autoencoder_list[-1].fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))
    else:
        autoencoder_list[-1].fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))
    encoded_imgs = autoencoder_list[0].predict(x_test)
    decoded_imgs = autoencoder_list[1].predict(encoded_imgs)
    autoencoded_imgs = autoencoder_list[2].predict(x_test)
    return (encoded_imgs, decoded_imgs, autoencoded_imgs), autoencoder_list

def plot_imgs_contrast(imgs, crow, name, show=True):
    fig = plt.figure()
    for i in range(crow):
        for j in range(len(imgs)):
            ax = plt.subplot(3, crow, i+1+j*crow)
            n = imgs[j][i].shape
            if np.prod(n) == 784:
                plt.imshow(imgs[j][i].reshape(28, 28))
            else:
                plt.imshow(imgs[j][i].reshape(n[0], np.prod(n[1:])))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig(name)
    if show:
        plt.show(block=False)
    return fig
    
def simplest_autoencoder_example(x_train=None, x_test=None, show_figure=True):
    if x_train == None:
        x_train, x_test = load_mnist_dataset()
    encoder, decoder, autoencoder = simplest_possible_autoencoder()
    imgs, autoencoder_list = train_autoencoder((encoder, decoder, autoencoder), x_train, x_test)
    plot_imgs_contrast((x_test, imgs[1], imgs[2]), 10, 'simplest_autoencoder.png', show=show_figure)
    
def sparsity_autoencoder_example(x_train=None, x_test=None, show_figure=True):
    if x_train == None:
        x_train, x_test = load_mnist_dataset()
    encoder, decoder, autoencoder = sparsity_constraint_autoencoder()
    imgs, autoencoder_list = train_autoencoder((encoder, decoder, autoencoder), x_train, x_test, optimizer='adadelta', epochs=100)
    plot_imgs_contrast((x_test, imgs[1], imgs[2]), 10, 'sparsity_autoencoder.png', show=show_figure)
    
def deep_autoencoder_example(x_train=None, x_test=None, show_figure=True):
    if x_train == None:
        x_train, x_test = load_mnist_dataset()
    encoder, decoder, autoencoder = deep_autoencoder()
    imgs, autoencoder_list = train_autoencoder((encoder, decoder, autoencoder), x_train, x_test, optimizer='adam', epochs=100)
    plot_imgs_contrast((x_test, imgs[1], imgs[2]), 10, 'deep_autoencoder.png', show=show_figure)
    
def convolutional_autoencoder_example(x_train=None, x_test=None, show_figure=True):
    if x_train == None:
        x_train, x_test = load_mnist_dataset(True)
    encoder, decoder, autoencoder = convolutional_autoencoder()
    imgs, autoencoder_list = train_autoencoder((encoder, decoder, autoencoder), x_train, x_test, epochs=10)
    plot_imgs_contrast(list(x_test)+list(imgs), 10, 'convolutional_autoencoder.png', show=show_figure)
    
def denosing_autoencoder_example(x_train=None, x_test=None, show_figure=True, noise_factor=0.5):
    if x_train == None:
        x_train, x_test = load_mnist_dataset(True)
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    plot_imgs_contrast((x_test, x_test_noisy), 10, show=show_figure)
    encoder, decoder, autoencoder = denosing_autoencoder()
    imgs, autoencoder_list = train_autoencoder((encoder, decoder, autoencoder), x_train_noisy, x_test_noisy, x_train, x_test, epochs=10)
    plot_imgs_contrast(list(x_test)+list(imgs), 10, 'denosing_autoencoder.png', show=show_figure)