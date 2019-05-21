import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i,x in enumerate(X):
      s = x.dot(W)
      s -= np.max(s)
      p = np.exp(s)/np.sum(np.exp(s))
      loss += -np.log(p[y[i]])

      grad_s = np.exp(s)/np.sum(np.exp(s))
      grad_s[y[i]] -= 1
      ####UGLY CODE
      x.shape= (3073,1)
      grad_s.shape = (1,10)
      dW += x.dot(grad_s)

  #divide by the number of examples
  loss /= X.shape[0]
  dW   /= X.shape[0]

  #L2 regularization
  loss += 0.5* reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_example = X.shape[0]

  S = X.dot(W)
  S -= np.amax(S, axis = 1,keepdims = True)
  P = np.exp(S)/np.sum(np.exp(S), axis = 1, keepdims = True)
  loss = np.sum(-np.log(np.choose(y,P.T)))

  P[np.arange(num_example), y] -= 1
  dW = X.T.dot(P)

  #divide by the number of examples
  loss /= num_example
  dW   /= num_example

  #L2 regularization
  loss += 0.5* reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

