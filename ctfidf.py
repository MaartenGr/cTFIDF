import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)
        self._idf_diag = None

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.

        """

        # Prepare input
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        # Calculate IDF scores
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=dtype)
        return self

    def transform(self, X: sp.csr_matrix, copy=True) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)

        """

        # Prepare input
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        # idf_ being a property, the automatic attributes detection
        # does not work as usual and we need to specify the attribute
        # name:
        check_is_fitted(self, attributes=["idf_"],
                        msg='idf vector is not fitted')

        # Check if expected nr features is found
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))

        X = X * self._idf_diag

        if self.norm:
            X = normalize(X, axis=1, norm='l1', copy=False)

        return X
