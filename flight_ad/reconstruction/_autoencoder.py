from sklearn.neural_network import MLPRegressor
from numpy import square

__all__ = ['AutoEncoderAD']


class AutoEncoder(MLPRegressor):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        """Init the AutoEncoder with default MPLRegressor inputs."""
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver=solver, alpha=alpha,
                         batch_size=batch_size, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init,
                         power_t=power_t, max_iter=max_iter, shuffle=shuffle,
                         random_state=random_state, tol=tol,
                         verbose=verbose, warm_start=warm_start, momentum=momentum,
                         nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                         epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    def fit(self, X, y=None):
        super().fit(X, X)

    def predict(self, X):
        return super().predict(X)


class AutoEncoderAD(AutoEncoder):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        """Init the AutoEncoderAD with default AutoEncoder inputs."""
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver=solver, alpha=alpha,
                         batch_size=batch_size, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init,
                         power_t=power_t, max_iter=max_iter, shuffle=shuffle,
                         random_state=random_state, tol=tol,
                         verbose=verbose, warm_start=warm_start, momentum=momentum,
                         nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                         epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        """Calculate samplewise squared error."""
        return square(X - super().predict(X)).mean(axis=1)


if __name__ == "__main__":
    autoencoder = AutoEncoder(
        hidden_layer_sizes=(500, 300, 2, 300, 500,),
        activation='tanh',
        solver='adam',
        learning_rate_init=1e-4,
        max_iter=20,
        tol=0.0000001,
        verbose=True
    )
