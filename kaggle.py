import pandas as pd
import numpy as np
import theano

def load_data():
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")
    zero_rows=train[train["TARGET"]==0]

    rbm_input=zero_rows.ix[:,1:-1]
    rbm_target=zero_rows.ix[:,-1]
    logistic_train_input=train.ix[:len(train)*0.7,1:-1]
    logistic_train_output=train.ix[:len(train)*0.7,-1]
    logistic_test_input=train.ix[len(train)*0.7:,1:-1]
    logistic_test_output=train.ix[len(train)*0.7:,-1]
    final_test_input=test.ix[:,1:]
    final_test_ids=test.ix[:,0]

    rbm = theano.shared(np.asarray(rbm_input.as_matrix(), dtype=theano.config.floatX)) , theano.shared(np.asarray(rbm_target.as_matrix(), dtype=theano.config.floatX))
    train_x= theano.shared(np.asarray(logistic_train_input.as_matrix(), dtype=theano.config.floatX)) ,theano.shared(np.asarray(logistic_train_output.as_matrix(), dtype=theano.config.floatX))
    test_x=theano.shared(np.asarray(logistic_test_input.as_matrix(), dtype=theano.config.floatX)) ,theano.shared(np.asarray( logistic_test_output.as_matrix(), dtype=theano.config.floatX))
    predict=theano.shared(np.asarray(final_test_input.as_matrix(), dtype=theano.config.floatX)) ,final_test_ids

    return rbm,train_x,test_x,predict
