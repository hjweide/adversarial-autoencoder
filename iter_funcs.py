def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size

    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        idx = slice(start, end)

        yield idx


if __name__ == '__main__':
    import numpy as np
    X = np.random.random((14, 4))
    print('Original data')
    print(X)
    print('As batches')
    for idx in get_batch_idx(X.shape[0], 4):
        print(X[idx])
