from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M) 120,3?
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    X = x.reshape((x.shape[0],w.shape[0])) # (N,D) = reshape((N, d_1, ..., d_k), N, D)
    # ^^^ force individual training samples into a single vector 
    out = X.dot(w) + b # (N,M) = (N,D).dot(D,M)
    # ^^^ dot multiply each sample with weight vector and add bias
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
#     import pdb; pdb.set_trace()
#     dx = np.reshape(w.dot(dout.T).T, (x.shape[0], x.shape[1], x.shape[2]))
#     # (N, d1, ..., d_k) = (D,M).dot((N,M).T).T # dL/dx = dL/dout * dout/dx ??? # we have to unpack D into d1, d2 etc.?
#     X = np.reshape(x,(x.shape[0],w.shape[0]))
#     dw = X.T.dot(dout) #, (w.shape[0], w.shape[1]))
#     # (D, M) = (N, d_1, ... d_k).dot((N, M))
#     db = np.sum(dout, axis=0)
    # (M,) = np.average((N, M), axis=0)
    
    dx = dout.dot(w.T).reshape(x.shape) 
    X = x.reshape((x.shape[0],w.shape[0]))
    dw = X.T.dot(dout) 
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
#     import pdb; pdb.set_trace()
    out = np.maximum(0,x)
#     out = x.maximum(0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x>0 * 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
#     import pdb; pdb.set_trace()
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    sample_mean = np.average(x,axis=0) 
    sample_var = np.var(x,axis=0)
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
#         N, D = x.shape

#         #step1: calculate mean
#         #mu = 1./N * np.sum(x, axis = 0)
#         mu = np.average(x,axis=0)

#         #step2: subtract mean vector of every trainings example
#         xmu = x
#         xmu -= mu

#         #step3: following the lower branch - calculation denominator
#         sq = xmu ** 2

#         #step4: calculate variance
#         #var = 1./N * np.sum(sq, axis = 0)
#         var = np.var(x,axis=0)

#         #step5: add eps for numerical stability, then sqrt
#         sqrtvar = np.sqrt(var + eps)

#         #step6: invert sqrtwar
#         ivar = 1./sqrtvar

#         #step7: execute normalization
#         #xhat = xmu * ivar
#         xhat = xmu
#         xhat /= sqrtvar

#         #step8: Nor the two transformation steps
#         gammax = gamma * xhat

#         #step9
#         out = gammax + beta

#         #store intermediate
#         cache['solution'] = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

        cache['eps'] = eps
        N, D = x.shape
        
#         # 1) compute sample mean and sample variance
#         #kratzert step1: calculate mean
#         mu = 1./N * np.sum(x, axis = 0)
        
#         #kratzert step2: subtract mean vector of every trainings example
#         xmu = x - mu

#         #kratzert step3: following the lower branch - calculation denominator
#         sq = xmu ** 2

#         #kratzert step4: calculate variance
#         var = 1./N * np.sum(sq, axis = 0)
        
        sample_mean = np.average(x,axis=0) 
        sample_var = np.var(x,axis=0)
        
        #import pdb; pdb.set_trace()
        # at this point the sample_mean === mu
        # and sample_var === var
        
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        
        
#         # 2) normalise the data with sample mean and sample variance
        
#         #kratzert step5: add eps for numerical stability, then sqrt
#         sqrtvar = np.sqrt(var + eps)

#         #kratzert step6: invert sqrtwar
#         ivar = 1./sqrtvar

#         #kratzert step7: execute normalization
#         xhat = xmu * ivar
    
        cache['x'] = x
#         x -= sample_mean
#         x /= np.sqrt(sample_var + eps)
        x_centered = x - sample_mean
        cache['x_centered'] = x_centered
        x_normalized = x_centered / np.sqrt(sample_var + eps)
        cache['norm_x'] = x_normalized
        
# #         import pdb; pdb.set_trace()
        
#         # 3) scale and shift the data with gamma and beta
        
#         #kratzert step8: Nor the two transformation steps
#         gammax = gamma * xhat

#         #kratzert step9
#         out = gammax + beta
        
#         cache['kratzert'] = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
#         import pdb; pdb.set_trace()
        out = gamma * x_normalized + beta # problem was using x here ...
#         x += beta 
        
        cache['gamma'] = gamma
        cache['beta'] = beta
        
#         out = x
        
        # corresponds to alg 1 on page 3 of the paper
        
#         running_mean = momentum * running_mean + (1 - momentum) * mu # was sample_mean
#         running_var = momentum * running_var + (1 - momentum) * var # was sample_var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        # 1) normalise the data with sample mean and sample variance
        
#         x -= running_mean
#         x /= np.sqrt(running_var)
        
        x_centered = x - sample_mean
        x_normalized = x_centered / np.sqrt(sample_var + eps)
        
#         import pdb; pdb.set_trace()
        
        # 2) scale and shift the data with gamma and beta
        
        out = gamma * x_normalized + beta 
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D) # (4,5)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    norm_x = cache['norm_x']
    gamma = cache['gamma']
    beta = cache['beta'] 
    eps = cache['eps'] 
    sample_mean = cache['sample_mean']
    sample_var = cache['sample_var']
    x_centered = cache['x_centered']
    
    #kratzert unfold the variables stored in cache
#     xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache['kratzert']
    
    
    N, D = norm_x.shape
    
#      #step9
#     dbeta = np.sum(dout, axis=0)
#     dgammax = dout #not necessary, but more understandable

#     #step8
#     dgamma = np.sum(dgammax*xhat, axis=0)
#     dxhat = dgammax * gamma

#     #step7
#     divar = np.sum(dxhat*xmu, axis=0)
#     dxmu1 = dxhat * ivar

#     #step6
#     dsqrtvar = -1. /(sqrtvar**2) * divar

#     #step5
#     dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

#     #step4
#     dsq = 1. /N * np.ones((N,D)) * dvar

#     #step3
#     dxmu2 = 2 * xmu * dsq

#     #step2
#     dx1 = (dxmu1 + dxmu2)
#     dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

#     #step1
#     dx2 = 1. /N * np.ones((N,D)) * dmu

#     #step0
#     dx = dx1 + dx2

    
    # step9
    dbeta = dout.sum(axis=0) # paper indicates this should be sum over dl/dy ==? dout
    # dgamma = (dout.T.dot(x)).sum(axis=0) # paper indicates this should be sum over dl/dy ==? dout * normalized x
    
    # step8
    dgamma = (dout*norm_x).sum(axis=0)
    # have tried both orderings and summing dout in advance ... none of them seem to work
    # is x (normalised x) the right value to be using here?
    
    dxhat = dout * gamma
    
    # step7
    divar = np.sum(dxhat*x_centered, axis=0) # missed this
    sqrtvar = np.sqrt(sample_var + eps)
    dcentered_mean = dxhat / sqrtvar
    
    # step6
    dsqrt_var = -1. /(sqrtvar**2) * divar # had these wrong way round
    
    # step5
    dvar = 0.5 * 1. / np.sqrt(sample_var + eps) * dsqrt_var
    
    # step4
    dx_centered_squared = 1. / N * np.ones((N, D)) * dvar
    
    # step3
    dx_centered = 2 * x_centered * dx_centered_squared
    
    # step2
    dx_1 = dx_centered + dcentered_mean
    dmean = -1 * (dx_centered + dcentered_mean).sum(axis=0)
    
    # step1 
    dx_2 = 1. / N * np.ones((N, D)) * dmean
    
    # step0
    dx = dx_1 + dx_2
    
    
    
    # mul gate is a gradient switcher, multiplying the incoming gradient by the other input ...
    
#     #unfold the variables stored in cache
#     xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache['solution']

#     #get the dimensions of the input/output
#     N,D = dout.shape

#     #step9
#     dbeta = np.sum(dout, axis=0)
#     dgammax = dout #not necessary, but more understandable

#     #step8
#     dgamma = np.sum(dgammax*xhat, axis=0)
#     dxhat = dgammax * gamma
    
    
#     import pdb; pdb.set_trace()
    
    
#     size = dout.shape[0]
#     d_norm = dout.dot(gamma)
#     d_sample_var = (d_norm.dot((x - sample_mean)).T * -1./2. * np.power(sample_var,-3./2)).sum(axis=0)
#     d_sample_mean =  (-1./np.sqrt(d_sample_var) * d_norm).sum(axis=0) + d_sample_var * (-2. * (x - sample_mean)).sum(axis=0) / size
    
#     dx = d_norm * 1./cache['sample_var'] + d_sample_mean * 2. * (cache['x'] - cache['sample_mean']) / size + d_sample_var / size
#     dx = dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    norm_x = cache['norm_x']
    N, D = norm_x.shape
    sample_var = cache['sample_var']
    sample_mean = cache['sample_mean']
    x_centered = cache['x_centered']
    eps = cache['eps'] 
    inv_sqrt_var = 1./np.sqrt(sample_var + eps)
    gamma = cache['gamma']
    beta = cache['beta'] 
    
    
    dbeta = dout.sum(axis=0)
    dgamma = (dout*norm_x).sum(axis=0)
    
    dnorm_x = dout * gamma
    dvar = (dnorm_x * x_centered * -1./2. * np.power(sample_var + eps,-3./2.)).sum(axis=0)
    dmean = (-1. * dnorm_x * inv_sqrt_var).sum(axis=0) + dvar / N * (-2. * x_centered).sum(axis=0)
    dx = dnorm_x * inv_sqrt_var + dvar * 2. * x_centered / N + dmean / N
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, {}
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    cache['eps'] = eps
    N, D = x.shape
    
    sample_mean = np.average(x,axis=1) 
    sample_var = np.var(x,axis=1)

    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    
    cache['x'] = x

#     import pdb; pdb.set_trace()
    x_centered = (x.T - sample_mean).T
    cache['x_centered'] = x_centered
    x_normalized = (x_centered.T / np.sqrt(sample_var + eps)).T
    cache['norm_x'] = x_normalized

    out = gamma * x_normalized + beta # problem was using x here ...

    cache['gamma'] = gamma
    cache['beta'] = beta

#     running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#     running_var = momentum * running_var + (1 - momentum) * sample_var
        
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
#     norm_x = cache['norm_x']
#     gamma = cache['gamma']
#     beta = cache['beta'] 
#     eps = cache['eps'] 
#     sample_mean = cache['sample_mean']
#     sample_var = cache['sample_var']
#     x_centered = cache['x_centered']
    
#     N, D = norm_x.shape
    
#     dbeta = dout.sum(axis=0)
#     dgamma = (dout*norm_x).sum(axis=0)
#     dxhat = dout * gamma
#     divar = np.sum(dxhat*x_centered, axis=1) 
#     sqrtvar = np.sqrt(sample_var + eps)
#     dcentered_mean = dxhat.T / sqrtvar
#     dsqrt_var = -1. /(sqrtvar**2) * divar 
#     dvar = 0.5 * 1. / np.sqrt(sample_var + eps) * dsqrt_var
# #     import pdb; pdb.set_trace()
#     dx_centered_squared = (1. / N * np.ones((N, D)).T * dvar).T
#     dx_centered = 2 * x_centered * dx_centered_squared
#     dx_1 = dx_centered.T + dcentered_mean
#     dmean = -1 * (dx_centered.T + dcentered_mean).sum(axis=0)
#     dx_2 = (1. / N * np.ones((N, D)).T * dmean).T
#     dx = (dx_1 + dx_2.T).T
    
    
    norm_x = cache['norm_x']
    N, D = norm_x.shape
    sample_var = cache['sample_var']
    sample_mean = cache['sample_mean']
    x_centered = cache['x_centered']
    eps = cache['eps'] 
    inv_sqrt_var = 1./np.sqrt(sample_var + eps)
    gamma = cache['gamma']
    beta = cache['beta'] 
    
    
    dbeta = dout.sum(axis=0)
    dgamma = (dout*norm_x).sum(axis=0)
    
    dnorm_x = dout * gamma
    # will need to re-do with writing out equations ...
#     import pdb; pdb.set_trace()
    dvar = ((dnorm_x * x_centered).T * -1./2. * np.power(sample_var + eps,-3./2.)).sum(axis=1)
    dmean = (-1. * dnorm_x.T * inv_sqrt_var).sum(axis=1) + dvar / N * (-2. * x_centered).sum(axis=0)
    dx = (dnorm_x.T * inv_sqrt_var).T + dvar * 2. * x_centered / N + dmean / N
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
