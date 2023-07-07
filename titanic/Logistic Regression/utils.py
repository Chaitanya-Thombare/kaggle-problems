import numpy as np
import pandas as pd

def forward_prop(Weights, Bias, X,):
    Z = np.dot(Weights.T, X) + Bias
    A = 1. / (1 + np.exp(-(Z)))
    return A

def prepare_results(training_records):

    column_set = pd.MultiIndex.from_tuples([
                                ('', 'Iteration'),
                                ('Training', 'Cost'),
                                ('Training', 'Accuracy'),
                                ('Training', 'Precision'), 
                                ('Training', 'Recall'),
                                ('Training', 'F1'),
                                ('Validation', 'Cost'),
                                ('Validation', 'Accuracy'),
                                ('Validation', 'Precision'), 
                                ('Validation', 'Recall'),
                                ('Validation', 'F1'),
                            ])

    training_records_df = pd.DataFrame(training_records, columns=column_set).set_index(('', 'Iteration'))
    training_records_df.index.names = ['Iteration']
    
    return training_records_df.round(2)
