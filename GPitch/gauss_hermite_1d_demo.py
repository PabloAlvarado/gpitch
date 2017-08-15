import numpy as np
import tensorflow as tf
import gpitch as gpi
from scipy import integrate
import quadratures as quad
import GPflow
import time


def variational_expectations(Fmu, Fvar, Y):
    H = 20 # get eval points and weights
    gh_x, gh_w = GPflow.quadrature.hermgauss(H)
    gh_x = gh_x.reshape(1, -1)
    gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)

    mean_f = Fmu[:, 0] # get mean and var of each q distribution, and reshape
    mean_g = Fmu[:, 1]
    var_f = Fvar[:, 0]
    var_g = Fvar[:, 1]
    Y, mean_f, mean_g, var_f, var_g = [tf.reshape(e, [-1, 1]) for e in (Y,
                                                                        mean_f,
                                                                        mean_g,
                                                                        var_f,
                                                                        var_g)]
    shape = tf.shape(mean_g) # get  output shape
    X = gh_x * tf.sqrt(2.*var_g) + mean_g # transformed evaluation points
    evaluations = 1. / (1. + tf.exp(-X)) # sigmoid function
    E1 =  tf.reshape(tf.matmul(evaluations, gh_w), shape) # compute expectations
    E2 =  tf.reshape(tf.matmul(evaluations**2, gh_w), shape)

    # finally! compute log-lik expectation under variational distribution
    var_exp = -0.5*((1./noise_var)*(Y**2 - 2.*Y*mean_f*E1 +
              (var_f + mean_f**2)*E2) + np.log(2.*np.pi) + np.log(noise_var))
    return var_exp

# generate random data, means and variances
N = 5
y = np.random.randn(N, 1)
q_mean = np.random.randn(N, 2)
q_var = np.random.rand(N, 2) # variance can only have positive values
noise_var = 0.01

# calculate variational expectations using 2 dimensional quadrature numpy
ground_t = np.zeros((N,1))
start_time = time.time()
for i in range(N):
    ground_t[i], _ = quad.ground_truth(y[i,0], q_mean[i][1], q_mean[i][0],
                                       q_var[i][1], q_var[i][0], noise_var)
print("--- %s seconds ---" % (time.time() - start_time))
print 'ground truth \n', ground_t, '\n'

# tensorflow implementation of variational expectations
Fmu  = tf.Variable(q_mean, dtype=tf.float64)
Fvar = tf.Variable(q_var, dtype=tf.float64)
Y = tf.Variable(y, dtype=tf.float64)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

start_time = time.time()
var_exp = variational_expectations(Fmu, Fvar, Y).eval(session=sess)
print("--- %s seconds --- " % (time.time() - start_time))
print 'variational_expectations \n', var_exp, '\n'







# # try it !!
# def variational_expectations_paad(self, Fmu, Fvar, Y):
#     H = 20 # get eval points and weights
#     gh_x, gh_w = GPflow.quadrature.hermgauss(H)
#     gh_x = gh_x.reshape(1, -1)
#     gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
#
#     mean_f = Fmu[:, 0] # get mean and var of each q distribution, and reshape
#     mean_g = Fmu[:, 1]
#     var_f = Fvar[:, 0]
#     var_g = Fvar[:, 1]
#     Y, mean_f, mean_g, var_f, var_g = [tf.reshape(e, [-1, 1]) for e in (Y,
#                                                                         mean_f,
#                                                                         mean_g,
#                                                                         var_f,
#                                                                         var_g)]
#     shape = tf.shape(mean_g) # get  output shape
#     X = gh_x * tf.sqrt(2.*var_g) + mean_g # transformed evaluation points
#     evaluations = 1. / (1. + tf.exp(-X)) # sigmoid function
#     E1 =  tf.reshape(tf.matmul(evaluations, gh_w), shape) # compute expectations
#     E2 =  tf.reshape(tf.matmul(evaluations**2, gh_w), shape)
#
#     # finally! compute log-lik expectation under variational distribution
#     var_exp = -0.5*((1./self.noise_var)*(Y**2 - 2.*Y*mean_f*E1 +
#               (var_f + mean_f**2)*E2) + np.log(2.*np.pi) + np.log(self.noise_var))
#     return var_exp





















#
