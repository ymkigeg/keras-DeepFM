import numpy as np
import random as rn
import tensorflow as tf
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Embedding, Dense, Flatten
from keras.layers import Concatenate, dot, Activation, Reshape
from keras.layers import BatchNormalization, concatenate, Dropout, Add
from keras.layers import RepeatVector, merge, Subtract, Lambda, Multiply
from keras.models import Model
from keras.regularizers import l2 as l2_reg
#from keras import initializations
import itertools
from keras import backend  as K
from keras.engine.topology import Layer
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

import os

np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


class MySumLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MySumLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            if K.ndim(x)!=K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0,2,1])
            x = x * mask
            if K.ndim(x)==2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)
        else:
            if K.ndim(x)==2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        if len(output_shape)==1:
            output_shape.append(1)
        return tuple(output_shape)

def getFileLineCount(filePath):
    rst = os.popen('wc -l ' + filePath)
    line = rst.readlines()[0]
    return int(line.strip().split(' ')[0])

def binary_crossentropy_with_ranking(y_true, y_pred):
    ''' Trying to combine ranking loss with numeric precision'''
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    # next, build a rank loss
    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))
    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(tf.boolean_mask(y_pred_score ,(y_true < 1)))
    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max
    # only keep losses for positive outcomes
    rankloss = tf.boolean_mask(rankloss,tf.equal(y_true,1))
    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))
    # average the loss for just the positive outcomes
    #tf.reduce_sum(tf.cast(myOtherTensor, tf.float32))
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(K.cast(y_true > 0, tf.float32) + 1))
    return (rankloss + 1) * logloss #- an alternative to try
    #return logloss

# PFA, prob false alert for binary classifier  
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = K.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = K.sum(y_pred - y_pred * y_true)  
    return FP/N 

# PTA prob true alerts for binary classifier  
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = K.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = K.sum(y_pred * y_true)  
    return TP/P

def auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return K.sum(s, axis=0)  

def log_loss(y_true, y_pred):
    ''' Trying to combine ranking loss with numeric precision'''
    # first get the log loss like normal
    logloss = K.sum(K.binary_crossentropy(y_true,y_pred), axis=-1)
    return logloss
    
class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, k=8, 
                dropout_keep_fm=[1.0, 1.0],
                deep_layers=[32, 32, 1], 
                dropout_keep_deep=[0.5, 0.5, 0.5],
                epoch=10, batch_size=256,
                learning_rate=0.001, optimizer_type='adam',
                verbose=1, random_seed=2016,
                use_fm=True, use_deep=True,
                loss_type='logloss', eval_metric='auc',
                l2=0.0, l2_fm=0.0, 
                log_dir = './output', bestModelPath = './output/keras.model',
                greater_is_better = True
                ):
        assert (use_fm or use_deep)
        assert loss_type in ['logloss', 'mse', 'ranking_logloss'], \
            'loss_type can be either "logloss" for classification task or "mse" for regression task or "ranking_logloss" for ranking task'

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.k = k    # denote as K, size of the feature embedding

        self.dropout_keep_fm = dropout_keep_fm
        self.deep_layers = deep_layers
        self.dropout_keep_deep = dropout_keep_deep

        self.use_fm = use_fm
        self.use_deep = use_deep

        self.verbose = verbose
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric

        self.l2 = l2
        self.l2_fm = l2_fm

        self.bestModelPath = bestModelPath
        self.log_dir = log_dir

        self.greater_is_better = greater_is_better

        self._init_graph()

    def _init_graph(self):

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
    
        self.feat_index = Input(shape=(self.field_size,)) #None*F
        self.feat_value = Input(shape=(self.field_size,)) #None*F
    
        self.embeddings = Embedding(self.feature_size, self.k, name='feature_embeddings', 
                embeddings_regularizer=l2_reg(self.l2_fm))(self.feat_index) #None*F*k
        feat_value = Reshape((self.field_size, 1))(self.feat_value) #None*F*1
        self.embeddings = Multiply()([self.embeddings, feat_value]) #None*F*8
    
        ###----first order------######
        self.y_first_order = Embedding(self.feature_size, 1, name='feature_bias', 
                embeddings_regularizer=l2_reg(self.l2))(self.feat_index) #None*F*1
        self.y_first_order = Multiply()([self.y_first_order, feat_value]) #None*F*1
        self.y_first_order = MySumLayer(axis=1)(self.y_first_order) # None*1
        self.y_first_order = Dropout(self.dropout_keep_fm[0], seed=self.seed)(self.y_first_order)
    
        ###------second order term-------###
        # sum_square part
        self.summed_feature_emb = MySumLayer(axis=1)(self.embeddings)                #None*k
        self.summed_feature_emb_squred = Multiply()([self.summed_feature_emb, self.summed_feature_emb]) #None*k
    
        # square_sum part
        self.squared_feature_emb = Multiply()([self.embeddings, self.embeddings])                 #None*F*k
        self.squared_sum_feature_emb = MySumLayer(axis=1)(self.squared_feature_emb)   #None*k
    
        # second order
        self.y_second_order = Subtract()([self.summed_feature_emb_squred, self.squared_sum_feature_emb])   #None*k
        self.y_second_order = Lambda(lambda x: x*0.5)(self.y_second_order)                      #None*k
        self.y_second_order = MySumLayer(axis=1)(self.y_second_order) #None*1
        self.y_second_order = Dropout(self.dropout_keep_fm[1], seed=self.seed)(self.y_second_order)
    
        ##deep 
        self.y_deep = Reshape((self.field_size * self.k,))(self.embeddings) # None*(F*k)

        for i in range(0, len(self.deep_layers)):
            self.y_deep = Dense(self.deep_layers[i], activation='relu')(self.y_deep)
            self.y_deep = Dropout(self.dropout_keep_deep[i], seed=self.seed)(self.y_deep) #None*32

        #deepFM
        if self.use_fm and self.use_deep:
            self.concat_y = Concatenate()([self.y_first_order, self.y_second_order, self.y_deep]) 
        elif self.use_fm:
            self.concat_y = Concatenate()([self.y_first_order, self.y_second_order]) 
        elif self.use_deep:
            self.concat_y = self.y_deep

        self.y = Dense(1, activation='sigmoid', name='main_output')(self.concat_y) #None*1
    
        self.model = Model(inputs=[self.feat_index, self.feat_value], outputs=self.y, name='model')
    
        if self.optimizer_type == 'adam':
            self.optimizer = Adam(lr=self.learning_rate, decay=0.1)

        if self.loss_type == 'ranking_logloss':
            self.loss = binary_crossentropy_with_ranking
            print('use ranking_logloss')
        elif self.loss_type == 'logloss':
            self.loss = 'binary_crossentropy'
            print('use logloss')
        elif self.loss_type == 'mse':
            self.loss = 'mean_squared_error'
            print('use mse')

        if self.eval_metric == 'auc':
            self.metrics = auc
        else:
            self.metrics = self.eval_metric

        self.model.compile(optimizer=self.optimizer,
                loss=self.loss, metrics=[self.metrics])
        

    def generate_data_on_libsvm(self, data_file):
        while 1:
            train_data = pd.read_csv(data_file, header=None, chunksize=self.batch_size, sep=' ')
            for data in train_data:
                batch_X = data.iloc[:,1:].values
                Xi = [ [ int(x.split(':')[0]) for x in line]  for line in batch_X]
                Xv = [ [ float(x.split(':')[1]) for x in line]  for line in batch_X]
                batch_y = data.iloc[:,0].values.reshape(-1, 1)
                yield ([np.array(Xi), np.array(Xv)], np.array(batch_y))


    def read_libsvm_data(self, data_file):
        valid_data = pd.read_csv(data_file, header=None, sep=' ')
        valid_X = valid_data.iloc[:,1:].values
        valid_Xi = [ [ int(x.split(':')[0]) for x in line]  for line in valid_X]
        valid_Xv = [ [ float(x.split(':')[1]) for x in line]  for line in valid_X]
        valid_y = valid_data.iloc[:,0].values.reshape(-1, 1)
        return ([np.array(valid_Xi), np.array(valid_Xv)], np.array(valid_y))


    def fit_on_libsvm(self, trainPath, validPath):
        monitor = 'val_' + self.eval_metric
        if self.greater_is_better:
            mode = 'max'
        else:
            mode = 'min'

        cb = [
          keras.callbacks.EarlyStopping(monitor=monitor, patience=5, verbose=self.verbose, mode=mode),
          keras.callbacks.ModelCheckpoint(self.bestModelPath, monitor=monitor, verbose=self.verbose, 
                        save_best_only=True, save_weights_only=False, mode=mode, period=1),
          keras.callbacks.TensorBoard(log_dir=self.log_dir), # histogram_freq=1), if validation_data is generator, can not use histogram
        ]

        total = getFileLineCount(trainPath)
        totalValid = getFileLineCount(validPath)
        print('total train samples: {}, total valid samples: {}'.format(total, totalValid))

        #validation_data = self.read_libsvm_data(validPath) 
        his = self.model.fit_generator(self.generate_data_on_libsvm(trainPath), 
                steps_per_epoch=total/self.batch_size, epochs=self.epoch, initial_epoch=0, 
                verbose=self.verbose, callbacks=cb,
                validation_data=self.generate_data_on_libsvm(validPath), validation_steps=totalValid/self.batch_size)



if __name__ == '__main__':

    trainPath = '../data/train_20190403.libsvm'
    validPath = '../data/valid_20190403.libsvm'
    indexPath = '../data/xgb_20190403_feature_mapping.csv'

    dfm_params = {
        'feature_size': 12026792,
        'field_size': 79,
        'k': 8,
        'use_fm': True,
        'use_deep': True,
        'dropout_keep_fm': [1.0, 1.0],
        'deep_layers': [32, 32, 1],
        'dropout_keep_deep': [0.5, 0.5, 0.5],
        'epoch': 30,
        'batch_size': 1024,
        'learning_rate': 0.01,
        'optimizer_type': 'adam',
        'verbose': 1,
        'random_seed': 1234,
        'loss_type': 'logloss',
        'eval_metric': 'auc',
        'l2': 0.01,
        'l2_fm': 0.01,
        'log_dir': '../keras_output',
        'bestModelPath': '../keras_output/keras.model',
        'greater_is_better': True
    }

    index_dfs = pd.read_csv(indexPath, index_col=False, sep='\t', chunksize=1000)
    feature_size = 0
    for index_df in index_dfs:
        # print(index_df.head())
        tmp = max(index_df['feat_id'])
        feature_size = tmp if feature_size < tmp else feature_size
    feature_size += 1
    dfm_params['feature_size'] = feature_size

    with open(trainPath, 'r') as f:
        line = f.readline()
        field_size = len(line.split(' ')) - 1
        dfm_params['field_size'] = field_size

    print('feature_size: {}, field_size: {}'.format(dfm_params['feature_size'], dfm_params['field_size']))

    dfm = DeepFM(**dfm_params)
    dfm.fit_on_libsvm(trainPath, validPath)

