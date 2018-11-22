import tensorflow as tf
import gpflow
from gpflow import params_as_tensors, name_scope


class ModGP(gpflow.models.Model):
    def __init__(self, X, Y, name=None):
        super().__init__(name=name)

        self.X = X.copy()
        self.Y = Y.copy()

        # # parameters
        self.a = gpflow.Param(1., gpflow.transforms.positive)
        self.b = gpflow.Param(1., gpflow.transforms.positive)

    # likelihood
    @params_as_tensors
    def _build_likelihood(self):

        # Get prior kl
        pass

        # Get conditionals
        pass

        # Get variational expectations
        pass

        # re-scale for minibatch size
        pass
        
        lik = self._build_predict(self.X)
        return -tf.reduce_sum(tf.abs(lik - self.Y))

    # prediction
    @params_as_tensors
    def _build_predict(self, Xnew):
        return tf.multiply(Xnew, self.a) + self.b

    # evaluate prediction
    @gpflow.params_as_tensors
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]))
    def predict_f(self, Xnew):
        return self._build_predict(Xnew)
