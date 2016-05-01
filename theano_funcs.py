import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.layers import get_all_params
from lasagne.updates import nesterov_momentum


# forward pass for the encoder, q(z|x)
def create_encoder_func(layers):
    X = T.fmatrix('X')
    X_batch = T.fmatrix('X_batch')

    Z = get_output(layers['l_encoder_out'], X, deterministic=True)

    encoder_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=Z,
        givens={
            X: X_batch,
        },
    )

    return encoder_func


# forward pass for the decoder, p(x|z)
def create_decoder_func(layers):
    Z = T.fmatrix('Z')
    Z_batch = T.fmatrix('Z_batch')

    X = get_output(
        layers['l_decoder_out'],
        inputs={
            layers['l_encoder_out']: Z
        },
        deterministic=True
    )

    decoder_func = theano.function(
        inputs=[theano.In(Z_batch)],
        outputs=X,
        givens={
            Z: Z_batch,
        },
    )

    return decoder_func


# forward/backward (optional) pass for the encoder/decoder pair
def create_encoder_decoder_func(layers, apply_updates=False):
    X = T.fmatrix('X')
    X_batch = T.fmatrix('X_batch')

    X_hat = get_output(layers['l_decoder_out'], X, deterministic=False)

    # reconstruction loss
    encoder_decoder_loss = T.mean(
        T.mean(T.sqr(X - X_hat), axis=1)
    )

    if apply_updates:
        # all layers that participate in the forward pass should be updated
        encoder_decoder_params = get_all_params(
            layers['l_decoder_out'], trainable=True)

        encoder_decoder_updates = nesterov_momentum(
            encoder_decoder_loss, encoder_decoder_params, 0.01, 0.9)
    else:
        encoder_decoder_updates = None

    encoder_decoder_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=encoder_decoder_loss,
        updates=encoder_decoder_updates,
        givens={
            X: X_batch,
        },
    )

    return encoder_decoder_func


# forward/backward (optional) pass for discriminator
def create_discriminator_func(layers, apply_updates=False):
    X = T.fmatrix('X')
    pz = T.fmatrix('pz')

    X_batch = T.fmatrix('X_batch')
    pz_batch = T.fmatrix('pz_batch')

    # the discriminator receives samples from q(z|x) and p(z)
    # and should predict to which distribution each sample belongs
    discriminator_outputs = get_output(
        layers['l_discriminator_out'],
        inputs={
            layers['l_prior_in']: pz,
            layers['l_encoder_in']: X,
        },
        deterministic=False,
    )

    # label samples from q(z|x) as 1 and samples from p(z) as 0
    discriminator_targets = T.vertical_stack(
        T.ones((X_batch.shape[0], 1)),
        T.zeros((pz_batch.shape[0], 1))
    )

    discriminator_loss = T.mean(
        T.nnet.binary_crossentropy(
            discriminator_outputs,
            discriminator_targets,
        )
    )

    if apply_updates:
        # only layers that are part of the discriminator should be updated
        discriminator_params = get_all_params(
            layers['l_discriminator_out'], trainable=True, discriminator=True)

        discriminator_updates = nesterov_momentum(
            discriminator_loss, discriminator_params, 0.1, 0.0)
    else:
        discriminator_updates = None

    discriminator_func = theano.function(
        inputs=[
            theano.In(X_batch),
            theano.In(pz_batch),
        ],
        outputs=discriminator_loss,
        updates=discriminator_updates,
        givens={
            X: X_batch,
            pz: pz_batch,
        },
    )

    return discriminator_func


# forward/backward (optional) pass for the generator
# note that the generator is the same network as the encoder,
# but updated separately
def create_generator_func(layers, apply_updates=False):
    X = T.fmatrix('X')
    X_batch = T.fmatrix('X_batch')

    # no need to pass an input to l_prior_in here
    generator_outputs = get_output(
        layers['l_encoder_out'], X, deterministic=False)

    # so pass the output of the generator as the output of the concat layer
    discriminator_outputs = get_output(
        layers['l_discriminator_out'],
        inputs={
            layers['l_prior_encoder_concat']: generator_outputs,
        },
        deterministic=False
    )

    # the discriminator learns to predict 1 for q(z|x),
    # so the generator should fool it into predicting 0
    generator_targets = T.zeros_like(X_batch.shape[0])

    # so the generator needs to push the discriminator's output to 0
    generator_loss = T.mean(
        T.nnet.binary_crossentropy(
            discriminator_outputs,
            generator_targets,
        )
    )

    if apply_updates:
        # only layers that are part of the generator (i.e., encoder)
        # should be updated
        generator_params = get_all_params(
            layers['l_discriminator_out'], trainable=True, generator=True)

        generator_updates = nesterov_momentum(
            generator_loss, generator_params, 0.1, 0.0)
    else:
        generator_updates = None

    generator_func = theano.function(
        inputs=[
            theano.In(X_batch),
        ],
        outputs=generator_loss,
        updates=generator_updates,
        givens={
            X: X_batch,
        },
    )

    return generator_func


if __name__ == '__main__':
    import model
    print('building model')
    layers = model.build_model()

    print('compiling theano functions')
    encoder_decoder_func = create_encoder_decoder_func(layers)
    discriminator_func = create_discriminator_func(layers)
    generator_func = create_generator_func(layers)

    import numpy as np
    X = np.random.random((16, 28 * 28)).astype(np.float32)
    pz = np.random.uniform(-2, 2, size=(16, 2)).astype(np.float32)

    print('X.shape = %r' % (X.shape,))
    print('pz.shape = %r' % (pz.shape,))

    print('running the three forward passes')
    print encoder_decoder_func(X)
    print discriminator_func(X, pz)
    print generator_func(X)
