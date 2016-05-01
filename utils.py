import numpy as np
from sklearn.utils import shuffle


# download mnist here: http://yann.lecun.com/exdb/mnist/
# g-unzip and copy to mnist directory
# inspiration:
# https://github.com/Newmu/dcgan_code/blob/master/mnist/load.py#L14
def load_mnist():
    with open('mnist/train-images-idx3-ubyte', 'rb') as f:
        data = np.fromfile(file=f, dtype=np.uint8)
    X_train = data[16:].reshape(60000, 28 * 28).astype(np.float32)
    with open('mnist/train-labels-idx1-ubyte', 'rb') as f:
        data = np.fromfile(file=f, dtype=np.uint8)
    y_train = data[8:].reshape(60000).astype(np.uint8)

    with open('mnist/t10k-images-idx3-ubyte', 'rb') as f:
        data = np.fromfile(file=f, dtype=np.uint8)
    X_test = data[16:].reshape(10000, 28 * 28).astype(np.float32)
    with open('mnist/t10k-labels-idx1-ubyte', 'rb') as f:
        data = np.fromfile(file=f, dtype=np.uint8)
    y_test = data[8:].reshape(10000).astype(np.uint8)

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    X_train /= 255.
    X_test /= 255.

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, _, X_test, _ = load_mnist()
    print X_train.shape, X_test.shape
    print X_train.min(), X_test.min()
    print X_train.mean(), X_test.mean()
    print X_train.max(), X_test.max()
