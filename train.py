# credit to https://github.com/dnouri/nolearn
# for inspiration when I was first learning to use Lasagne

import model
import theano_funcs
import utils

from iter_funcs import get_batch_idx

import numpy as np
from lasagne.layers import get_all_param_values
from os.path import join


def train_autoencoder():
    print('building model')
    layers = model.build_model()

    max_epochs = 5000
    batch_size = 128
    weightsfile = join('weights', 'weights_train_val.pickle')

    print('compiling theano functions for training')
    print('  encoder/decoder')
    encoder_decoder_update = theano_funcs.create_encoder_decoder_func(
        layers, apply_updates=True)
    print('  discriminator')
    discriminator_update = theano_funcs.create_discriminator_func(
        layers, apply_updates=True)
    print('  generator')
    generator_update = theano_funcs.create_generator_func(
        layers, apply_updates=True)

    print('compiling theano functions for validation')
    print('  encoder/decoder')
    encoder_decoder_func = theano_funcs.create_encoder_decoder_func(layers)
    print('  discriminator')
    discriminator_func = theano_funcs.create_discriminator_func(layers)
    print('  generator')
    generator_func = theano_funcs.create_generator_func(layers)

    print('loading data')
    X_train, y_train, X_test, y_test = utils.load_mnist()

    try:
        for epoch in range(1, max_epochs + 1):
            print('epoch %d' % (epoch))

            # compute loss on training data and apply gradient updates
            train_reconstruction_losses = []
            train_discriminative_losses = []
            train_generative_losses = []
            for train_idx in get_batch_idx(X_train.shape[0], batch_size):
                X_train_batch = X_train[train_idx]
                # 1.) update the encoder/decoder to min. reconstruction loss
                train_batch_reconstruction_loss =\
                    encoder_decoder_update(X_train_batch)

                # sample from p(z)
                pz_train_batch = np.random.uniform(
                    low=-2, high=2,
                    size=(X_train_batch.shape[0], 2)).astype(
                        np.float32)

                # 2.) update discriminator to separate q(z|x) from p(z)
                train_batch_discriminative_loss =\
                    discriminator_update(X_train_batch, pz_train_batch)

                # 3.)  update generator to output q(z|x) that mimic p(z)
                train_batch_generative_loss = generator_update(X_train_batch)

                train_reconstruction_losses.append(
                    train_batch_reconstruction_loss)
                train_discriminative_losses.append(
                    train_batch_discriminative_loss)
                train_generative_losses.append(
                    train_batch_generative_loss)

            # average over minibatches
            train_reconstruction_losses_mean = np.mean(
                train_reconstruction_losses)
            train_discriminative_losses_mean = np.mean(
                train_discriminative_losses)
            train_generative_losses_mean = np.mean(
                train_generative_losses)

            print('  train: rec = %.6f, dis = %.6f, gen = %.6f' % (
                train_reconstruction_losses_mean,
                train_discriminative_losses_mean,
                train_generative_losses_mean,
            ))

            # compute loss on test data
            test_reconstruction_losses = []
            test_discriminative_losses = []
            test_generative_losses = []
            for test_idx in get_batch_idx(X_test.shape[0], batch_size):
                X_test_batch = X_test[test_idx]
                test_batch_reconstruction_loss =\
                    encoder_decoder_func(X_test_batch)

                # sample from p(z)
                pz_test_batch = np.random.uniform(
                    low=-2, high=2,
                    size=(X_test.shape[0], 2)).astype(
                        np.float32)

                test_batch_discriminative_loss =\
                    discriminator_func(X_test_batch, pz_test_batch)

                test_batch_generative_loss = generator_func(X_test_batch)

                test_reconstruction_losses.append(
                    test_batch_reconstruction_loss)
                test_discriminative_losses.append(
                    test_batch_discriminative_loss)
                test_generative_losses.append(
                    test_batch_generative_loss)

            test_reconstruction_losses_mean = np.mean(
                test_reconstruction_losses)
            test_discriminative_losses_mean = np.mean(
                test_discriminative_losses)
            test_generative_losses_mean = np.mean(
                test_generative_losses)

            print('  test: rec = %.6f, dis = %.6f, gen = %.6f' % (
                test_reconstruction_losses_mean,
                test_discriminative_losses_mean,
                test_generative_losses_mean,
            ))

    except KeyboardInterrupt:
        print('caught ctrl-c, stopped training')
        weights = get_all_param_values([
            layers['l_decoder_out'],
            layers['l_discriminator_out'],
        ])
        print('saving weights to %s' % (weightsfile))
        model.save_weights(weights, weightsfile)


if __name__ == '__main__':
    train_autoencoder()
