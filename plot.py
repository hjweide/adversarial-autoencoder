import model
import theano_funcs
import utils

from iter_funcs import get_batch_idx

# credit to @fulhack: https://twitter.com/fulhack/status/721842480140967936
import seaborn  # NOQA - never used, but improves matplotlib's style
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.decomposition import PCA

from os.path import join


def plot(Z1, y1, Z2, y2, filename=None, title=None):
    digit_colors = [
        'red', 'green', 'blue', 'cyan', 'magenta',
        'yellow', 'black', 'white', 'orange', 'gray',
    ]

    legend, labels = [], []
    for i in range(0, 10):
        idx1 = y1 == i
        idx2 = y2 == i
        pc1 = plt.scatter(
            Z1[idx1, 0], Z1[idx1, 1],
            marker='o', color=digit_colors[i],
        )
        legend.append(pc1)
        labels.append('%d' % i)
        pc2 = plt.scatter(
            Z2[idx2, 0], Z2[idx2, 1],
            marker='x', color=digit_colors[i],
        )
        legend.append(pc2)
        labels.append('%d' % i)

    # only plot digit colors to avoid cluttering the legend
    plt.legend(legend[::2], labels[::2], loc='upper left', ncol=1)
    if title is not None:
        plt.title(title)
    if filename is None:
        filename = 'plot.png'
    plt.savefig(filename, bbox_inches='tight')


# always a good sanity check
def plot_pca():
    print('loading data')
    X_train, y_train, X_test, y_test = utils.load_mnist()
    pca = PCA(n_components=2)

    print('transforming training data')
    Z_train = pca.fit_transform(X_train)

    print('transforming test data')
    Z_test = pca.transform(X_test)

    plot(Z_train, y_train, Z_test, y_test,
         filename='pca.png', title='projected onto principle components')


def plot_autoencoder(weightsfile):
    print('building model')
    layers = model.build_model()

    batch_size = 128

    print('compiling theano function')
    encoder_func = theano_funcs.create_encoder_func(layers)

    print('loading weights from %s' % (weightsfile))
    model.load_weights([
        layers['l_decoder_out'],
        layers['l_discriminator_out'],
    ], weightsfile)

    print('loading data')
    X_train, y_train, X_test, y_test = utils.load_mnist()

    train_datapoints = []
    print('transforming training data')
    for train_idx in get_batch_idx(X_train.shape[0], batch_size):
        X_train_batch = X_train[train_idx]
        train_batch_codes = encoder_func(X_train_batch)
        train_datapoints.append(train_batch_codes)

    test_datapoints = []
    print('transforming test data')
    for test_idx in get_batch_idx(X_test.shape[0], batch_size):
        X_test_batch = X_test[test_idx]
        test_batch_codes = encoder_func(X_test_batch)
        test_datapoints.append(test_batch_codes)

    Z_train = np.vstack(train_datapoints)
    Z_test = np.vstack(test_datapoints)

    plot(Z_train, y_train, Z_test, y_test,
         filename='adversarial_train_val.png',
         title='projected onto latent space of autoencoder')


def plot_latent_space(weightsfile):
    print('building model')
    layers = model.build_model()
    batch_size = 128
    decoder_func = theano_funcs.create_decoder_func(layers)

    print('loading weights from %s' % (weightsfile))
    model.load_weights([
        layers['l_decoder_out'],
        layers['l_discriminator_out'],
    ], weightsfile)

    # regularly-spaced grid of points sampled from p(z)
    Z = np.mgrid[2:-2.2:-0.2, -2:2.2:0.2].reshape(2, -1).T[:, ::-1].astype(np.float32)

    reconstructions = []
    print('generating samples')
    for idx in get_batch_idx(Z.shape[0], batch_size):
        Z_batch = Z[idx]
        X_batch = decoder_func(Z_batch)
        reconstructions.append(X_batch)

    X = np.vstack(reconstructions)
    X = X.reshape(X.shape[0], 28, 28)

    fig = plt.figure(1, (12., 12.))
    ax1 = plt.axes(frameon=False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.title('samples generated from latent space of autoencoder')
    grid = ImageGrid(
        fig, 111, nrows_ncols=(21, 21),
        share_all=True)

    print('plotting latent space')
    for i, x in enumerate(X):
        img = (x * 255).astype(np.uint8)
        grid[i].imshow(img, cmap='Greys_r')
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        grid[i].set_frame_on(False)

    plt.savefig('latent_train_val.png', bbox_inches='tight')


if __name__ == '__main__':
    weightsfile = join('weights', 'weights_train_val.pickle')
    #plot_autoencoder(weightsfile)
    #plot_pca()
    plot_latent_space(weightsfile)
