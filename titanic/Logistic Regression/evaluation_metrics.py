import numpy as np
import pandas as pd

def binary_accuracy(y_p, y_g):    
    y_p = y_p > 0.5
    return np.mean((y_p == y_g))

def confusion_matrix(y_p, y_g):
    y_p = y_p > 0.5

    conf_mtx = pd.DataFrame(columns=['True', 'False'], index=['Positive', 'Negative'])

    P_idx = y_g == 1
    N_idx = y_g == 0

    conf_mtx['True']['Positive'] = np.sum(y_g[P_idx] == y_p[P_idx])
    conf_mtx['True']['Negative'] = np.sum(y_g[N_idx] == y_p[N_idx])
    conf_mtx['False']['Positive'] = np.sum(y_g[N_idx] != y_p[N_idx])
    conf_mtx['False']['Negative'] = np.sum(y_g[P_idx] != y_p[P_idx])

    if (conf_mtx['True']['Positive'] == 0 or conf_mtx['False']['Positive'] == 0): precision = 0
    else: precision = conf_mtx['True']['Positive'] / (conf_mtx['True']['Positive'] + conf_mtx['False']['Positive'])
    
    if (conf_mtx['True']['Positive'] == 0 or conf_mtx['False']['Negative'] == 0): recall = 0
    else: recall = conf_mtx['True']['Positive'] / (conf_mtx['True']['Positive'] + conf_mtx['False']['Negative'])

    if precision == recall == 0: f1 = 0
    else: f1 = 2 * precision * recall / (precision + recall)

    return conf_mtx, precision, recall, f1

