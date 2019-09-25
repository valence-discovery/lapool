from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin
import numpy as np


class LogNormScaler(MinMaxScaler):
    r"""
    Log normal scaler. This transformer first scales the input into a range 
    between [0-1], then applies a log-scaling on the resulting output.
    """

    def fit(self, X, y=None):
        res = super().fit(X, y)
        return res

    def transform(self, X):
        res = super().transform(X)
        return np.log(res + 1e-8)  # trying to avoid log (0)

    def inverse_transform(self, X):
        res = np.exp(X)
        return super().inverse_transform(res)


class NoneScaler(TransformerMixin):
    r"""
    No scaler. This transformer does nothing. It is offered for convenience
    so the same API can be used as with other scalers
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def get_scaler(scname):
    r"""
    Get a scaler transformer by name. Currently supported scalers
    are `lognorm`, `norm`, `minmax`, and `none`

    Arguments
    ----------
        scname: str
            the name of the scaler to return

    Returns
    -------
        scaler: An instance of a scaler class that offers, fit and transform methods.
    """
    SCALERS = {"lognorm": LogNormScaler(), "norm": StandardScaler(),
               "minmax": MinMaxScaler(), 'none': NoneScaler()}
    return SCALERS[scname.lower()]
