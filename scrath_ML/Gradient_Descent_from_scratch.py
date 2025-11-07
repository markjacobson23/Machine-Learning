import numpy as np


def gradient_descent_from_scratch(x, y, batch, shuffle=False, lr=.001, tol=1e-8, max_epochs=10000, verbose=False):
    """this is an implementation of linear regression using batch gradient descent written from scratch with NumPy."""
    def mse(yhat, y):
        """Computes mean square error between prediction and target, scaled
        by 1/2 and length of target."""
        loss = (.5 / len(y)) * (np.linalg.norm(y - yhat) ** 2)
        return loss

    def grad(xb, yb, w):
        """Computes vector gradient of the mse function evaluated at the current weights. """
        return (-1 / xb.shape[0]) * np.linalg.matrix_transpose(xb) @ (yb - xb @ w)

    w = np.zeros(x.shape[1], dtype=float)
    prev = np.inf
    for epoch in range(max_epochs):
        idx = np.arange(x.shape[0])
        if shuffle:
            idx = np.random.permutation(x.shape[0])
        for start in range(0, x.shape[0], batch):
            batch_idx = idx[start:start+batch]
            xb, yb = x[batch_idx], y[batch_idx]
            w -= lr * grad(xb, yb, w)
        loss = mse(x @ w, y)
        if verbose and epoch % 1000 == 0:
            print(f'epoch {epoch}   Loss {loss:.6f}')
        if abs(loss-prev) < tol:
            break
        prev = loss
    return w
