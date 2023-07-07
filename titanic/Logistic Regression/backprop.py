import numpy as np

momentum_var, rms_prop_var = {}, {}

def gradient_descent(pred_Y, grou_Y, Weights, Bias, X, hyp):
    cost_f = hyp['cost_function']
    cost = cost_f(pred_Y.T, grou_Y.T)
    dZ = pred_Y - grou_Y
    dW = np.dot(X, dZ.T) / hyp['m_train']
    db = np.sum(dZ, axis=1, keepdims=True) / hyp['m_train']
    Weights -= dW * hyp['learning_rate']
    Bias -= db * hyp['learning_rate']

    return cost, Weights, Bias

def momentum_gradient_descent(pred_Y, grou_Y, Weights, Bias, X, hyp):
    cost_f = hyp['cost_function']
    cost = cost_f(pred_Y.T, grou_Y.T)

    dZ = pred_Y - grou_Y
    dW = np.dot(X, dZ.T) / hyp['m_train']
    db = np.sum(dZ, axis=1, keepdims=True) / hyp['m_train']

    if not momentum_var:
        momentum_var['vdW'], momentum_var['vdb'] = dW, db
    
    vdW = (hyp['beta1'] * momentum_var['vdW'] + (1 - hyp['beta1']) * dW)
    vdb = (hyp['beta1'] * momentum_var['vdb'] + (1 - hyp['beta1']) * db)
    
    momentum_var['vdW'], momentum_var['vdb'] = vdW, vdb

    Weights -= vdW * hyp['learning_rate']
    Bias -= vdb * hyp['learning_rate']

    return cost, Weights, Bias

def RMSprop_gradient_descent(pred_Y, grou_Y, Weights, Bias, X, hyp):
    cost_f = hyp['cost_function']
    cost = cost_f(pred_Y.T, grou_Y.T)

    dZ = pred_Y - grou_Y
    dW = np.dot(X, dZ.T) / hyp['m_train']
    db = np.sum(dZ, axis=1, keepdims=True) / hyp['m_train']

    if not rms_prop_var:
        rms_prop_var['sdW'], rms_prop_var['sdb'] = abs(dW), abs(db)

    sdW = (hyp['beta2'] * rms_prop_var['sdW'] + (1 - hyp['beta2']) * (dW ** 2))
    sdb = (hyp['beta2'] * rms_prop_var['sdb'] + (1 - hyp['beta2']) * (db ** 2))
    
    rms_prop_var['sdW'], rms_prop_var['sdb'] = sdW, sdb

    Weights -= hyp['learning_rate'] * dW / np.sqrt(sdW)
    Bias -= hyp['learning_rate'] * db / np.sqrt(sdb)

    return cost, Weights, Bias

def ADAM(pred_Y, grou_Y, Weights, Bias, X, hyp):
    raise Exception("Sorry, Adam is under development")