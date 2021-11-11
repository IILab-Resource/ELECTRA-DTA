#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os


import sys
import keras


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional,Multiply 
# Merge, 
from keras.layers import BatchNormalization, merge, add
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.optimizers import Adam,  RMSprop

import keras.backend.tensorflow_backend as KTF

import numpy as np
from tqdm import tqdm
from keras.layers import Input, CuDNNGRU, GRU
from numpy import linalg as LA
import scipy
#from sklearn.model_selection import KFold, ShuffleSplit
from keras import backend as K
import re
#from multiHead import  SparseSelfAttention

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

KTF.set_session(sess)


# In[2]:


hidden_dim = 256 #256

from six.moves import cPickle as pickle #for performance


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


# In[ ]:


all_protein_seqs_emb = []
all_smiles_seqs_emb = []

EMB_NO = 12


for i in range(1,EMB_NO+1):
    if i < 10:
        embedding_no = '0'+str(i)
    else:
        embedding_no = i

    protein_seqs_emb  = load_dict('dataset/embedding256-12layers/atomwise_BindingDB-full_protein_maxlen1022_dim256-layer{}.pkl'.format(embedding_no))
    smiles_seqs_emb = load_dict('dataset/embedding256-12layers/atomwise_BindingDB-full_smiles_maxlen100_dim256-layer{}.pkl'.format(embedding_no))
    all_protein_seqs_emb.append(protein_seqs_emb)
    all_smiles_seqs_emb.append(smiles_seqs_emb)
    

def dict_mean(all_emb):
    sums = Counter()
    counters = Counter()
    for itemset in all_emb:
        sums.update(itemset)
        counters.update(itemset.keys())

    ret = {x: sums[x]/counters[x] for x in sums.keys()}
    return ret
    
    
from collections import Counter


protein_mean_emb = dict_mean(all_protein_seqs_emb)
smiles_mean_emb = dict_mean(all_smiles_seqs_emb)

   
    


# In[ ]:


def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select


# In[4]:
def load_emb_from_dict(emb_dict, key, max_len):
    
    X = np.zeros(( max_len,hidden_dim  ))
    emb = emb_dict[key]
             
    emb_shape = emb.shape[0]
    if emb_shape > max_len:
        X = emb[:max_len]
    else:
        X[:emb_shape,:] = emb
        
    return X
    

import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, prots, drugs, Y, batch_size=256):
        'Initialization'
        self.batch_size = batch_size
        self.prots = prots
        self.drugs = drugs
        self.Y = Y
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.prots) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.prots))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        input_list = []
         
        X_drug = np.zeros((self.batch_size, smilen,hidden_dim))
        X_prot_seq = np.zeros((self.batch_size, seq_len,hidden_dim))


        for i, ID in enumerate(list_IDs_temp):
            X_drug[i] = load_emb_from_dict(smiles_mean_emb, self.drugs[ID], smilen)
            X_prot_seq[i] = load_emb_from_dict(protein_mean_emb, self.prots[ID], seq_len)

        input_list.append(X_drug)
        input_list.append(X_prot_seq)
            
                    
            
            
 
        y = np.zeros((self.batch_size))
    
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.Y[ID]

        return input_list ,  y


# In[ ]:


def Highway(value, n_layers, activation="tanh", gate_bias=0):  
    """ Highway layers:
        a minus bias means the network is biased towards carry behavior in the initial stages"""
    dim = K.int_shape(value)[-1]
    bias = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):
        T_gate = Dense(units=dim, bias_initializer=bias, activation="sigmoid")(value)
        C_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(T_gate)
        transform = Dense(units=dim, activation=activation)(value)
        transform_gated = Multiply()([T_gate, transform])
        carry_gated = Multiply()([C_gate, value])
        value = Add()([transform_gated, carry_gated])
    return value


# In[7]:


#from keras_radam import RAdam
#from keras_lookahead import Lookahead
from keras.layers import Lambda,Add, CuDNNGRU,TimeDistributed, Bidirectional,Softmax
from keras import regularizers
from keras.regularizers import l2
import tensorflow as tf
from keras import regularizers
from sklearn.model_selection import KFold, ShuffleSplit

smilen = 100
seq_len = 1000
# Squeeze and Excitation
def se_block(input, channels, r=8):
    # Squeeze
    x = GlobalAveragePooling1D()(input)
    # Excitation
    x = Dense(channels//r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])

def coeff_fun_prot(x):
    import tensorflow as tf
    import keras
        
    tmp_a_1=tf.keras.backend.mean(x[0],axis=-1,keepdims=True)
    tmp_a_1=tf.nn.softmax(tmp_a_1)
    tmp=tf.tile(tmp_a_1,(1,1,keras.backend.int_shape(x[1])[2]))
    return tf.multiply(x[1],tmp)
    
def att_func(x):
    import tensorflow as tf
    import keras
    
    tmp_a_2=tf.keras.backend.permute_dimensions(x[1],(0,2,1))
    mean_all=tf.keras.backend.sigmoid(tf.keras.backend.batch_dot(tf.keras.backend.mean(x[0],axis=1,keepdims=True),tf.keras.backend.mean(tmp_a_2,axis=-1,keepdims=True)))
    tmp_a=tf.keras.backend.sigmoid(tf.keras.backend.batch_dot(x[0],tmp_a_2))*mean_all
    #tmp_a=tf.nn.softmax(tmp_a)
    return tmp_a
def coeff_fun_lig(x):
    import tensorflow as tf
    import keras
    tmp1=tf.keras.backend.permute_dimensions(x[0],(0,2,1))
    tmp_a_1=tf.keras.backend.mean(tmp1,axis=-1,keepdims=True)
    tmp_a_1=tf.nn.softmax(tmp_a_1)
    tmp=tf.tile(tmp_a_1,(1,1,keras.backend.int_shape(x[1])[2]))
    return tf.multiply(x[1],tmp)

def conv_block(inputs, seblock, NUM_FILTERS,FILTER_LENGTH1):
     
        
    conv1_encode = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,   activation='relu', padding='valid', strides=1)(inputs)
    if seblock: 
        conv1_encode = se_block(conv1_encode,NUM_FILTERS)
    
    conv2_encode = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid', strides=1)(conv1_encode)
    if seblock: 
        conv2_encode = se_block(conv2_encode,NUM_FILTERS*2) 
    
    return conv2_encode



    

    
def fc_net(encode_interaction):
    n_layers = 4
    gate = Highway(n_layers = n_layers, value=encode_interaction, gate_bias=-2)
    FC1 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(gate)
    FC2 = Dropout(0.4)(FC1)
    FC2 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(FC2)
    FC2 = Dropout(0.4)(FC2)
    FC2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(FC2)
#     FC2 = Dropout(0.3)(FC2) 
    
    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2)
    return predictions
def share_conv_block(protein_conv1_encode, protein_conv2_encode,comp_conv1_encode,comp_conv2_encode,prot_emb, comp_emb):
    prot_emb = protein_conv1_encode(prot_emb)
    prot_emb = protein_conv2_encode(prot_emb)
    
    comp_emb = comp_conv1_encode(comp_emb)
    comp_emb = comp_conv2_encode(comp_emb)
    
    encode_protein = GlobalMaxPooling1D()(prot_emb)
    encode_smiles = GlobalMaxPooling1D()(comp_emb) 
    

    encode_interaction =   Concatenate()([encode_smiles, encode_protein])
    
    predictions = fc_net(encode_interaction)
    
    return predictions
    
def FFN(inputs):
    encode = Dense(256, activation='relu')(inputs)
    encode = Dense(256)(encode)
    return encode    
    
def build_model():
    
    
    drugInput = Input(shape=(smilen,hidden_dim))
    protInput = Input(shape=(seq_len,hidden_dim))
    
    # share CNN
    NUM_FILTERS = hidden_dim
    
    FILTER_LENGTH1 = 3
    
    n_layers = 4
    seblock = True
    
#     encode_prot = FFN(protInput)
#     encode_smiles = FFN(drugInput)
    
# #     att_tmp=TimeDistributed(Dense(hidden_dim,use_bias=False))(encode_prot)
#     att=Lambda(att_func)([encode_prot,encode_smiles])
#     encode_prot=Lambda(coeff_fun_prot)([att,encode_prot])
#     encode_smiles=Lambda(coeff_fun_lig)([att,encode_smiles])
    

    
    encode_smiles = conv_block(drugInput,seblock, NUM_FILTERS, 3)
    encode_prot = conv_block(protInput,seblock, NUM_FILTERS, 3)
    
   
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)
    encode_prot = GlobalMaxPooling1D()(encode_prot)
    

    
    
    encode_interaction =   Concatenate()([encode_smiles, encode_prot])


#     gate = Highway(n_layers = n_layers, value=encode_interaction, gate_bias=-2)
    
    
    predictions = fc_net(encode_interaction)
    
    # And add a logistic regression on top
#     predictions = Dense(1, kernel_initializer='normal')(gate) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs= [drugInput,protInput ], outputs=[predictions])
   # adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
#     ranger =  Lookahead(RAdam())
    interactionModel.compile(optimizer= 'adam', loss='mse', metrics=[cindex_score]) #, metrics=['cindex_score']
    return interactionModel


model = build_model()
print(model.summary())


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.callbacks import TensorBoard
#from sklearn.metrics import mean_squared_error
from rlscore.measure import cindex
from sklearn.model_selection import KFold

from emtrics import *
all_loss =  np.zeros((5,1))
all_ci =  np.zeros((5,1))
all_ci2 = np.zeros((5,1))
all_mse2 = np.zeros((5,1))
all_r = np.zeros((5,1))
all_aupr = np.zeros((5,1))
all_rm2 = np.zeros((5,1))

data_file = 'dataset/BindingDB-uniq-data.csv'

all_drug = []
all_protein = []
all_Y = []



with open(data_file, 'r') as f:
    all_lines = f.readlines()
    
for line in all_lines:
        row = line.rstrip().split(',')
        all_drug.append(row[0])
        all_protein.append(row[1])
        all_Y.append(float(row[2]))

print(len(all_Y), len(all_drug), len(all_protein))

batch_size = 256
# set random_state as 
kf = KFold(n_splits=5, shuffle=True)
for split, ( train_index, test_index) in enumerate( kf.split(all_Y)):
    print(train_index,test_index )

    train_protein_cv = np.array(all_protein)[train_index]
    train_drug_cv = np.array(all_drug)[train_index]
    train_Y_cv = np.array(all_Y)[train_index]
    test_protein_cv = np.array(all_protein)[test_index]
    test_drug_cv = np.array(all_drug)[test_index]
    test_Y_cv = np.array(all_Y)[test_index]

    train_size = train_protein_cv.shape[0]  
    
    valid_size = 0 #int(len(all_Y)/5.0) # 7051 #?
    training_generator = DataGenerator( train_protein_cv[:train_size-valid_size], train_drug_cv[:train_size-valid_size],
                                       np.array(train_Y_cv[:train_size-valid_size]),batch_size=batch_size)
#     validate_generator = DataGenerator( train_protein_cv[train_size-valid_size:], 
#                                        train_drug_cv[train_size-valid_size:], 
#                                        np.array(train_Y_cv[train_size-valid_size:]),batch_size=batch_size)

    save_model_name = 'models-bdbki-embedding-avg'+str(split)
    
    model = build_model()
     
    save_checkpoint = ModelCheckpoint(save_model_name, verbose=1,save_best_only=True, monitor='loss', save_weights_only=True, mode='min') 
    earlyStopping = EarlyStopping(monitor='loss',  patience=25, verbose=1,mode='min')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
    model.fit_generator(generator=training_generator,      
                        epochs = 500 ,
                        verbose=1,callbacks=[earlyStopping, save_checkpoint])

    
#     model.fit_generator(generator=training_generator,      
#                         epochs = 500 ,
#                         verbose=1, validation_data=validate_generator,
#                         callbacks=[earlyStopping, save_checkpoint])
    
    
    
    
    input_list = []
    
    X_drug = np.zeros((len(test_drug_cv), smilen,hidden_dim))
    X_prot_seq = np.zeros((len(test_protein_cv), seq_len,hidden_dim))


    for i in range(len(test_protein_cv)):
        X_drug[i] = load_emb_from_dict(smiles_mean_emb, test_drug_cv[i], smilen)
        X_prot_seq[i] = load_emb_from_dict(protein_mean_emb, test_protein_cv[i], seq_len)

    input_list.append(X_drug)
    input_list.append(X_prot_seq)
    
    
    


    model.load_weights(save_model_name)

    y_pred = model.predict(input_list)
    
    test_Y_cv = np.float64(np.array(test_Y_cv)) 
    
    y_pred = np.float64(np.array(y_pred)) 
    
    ci2 = cindex(test_Y_cv, y_pred)
    
    rm2 = get_rm2(test_Y_cv, y_pred[:,0])   
    mse = get_mse(test_Y_cv, y_pred[:,0])
    pearson = get_pearson(test_Y_cv, y_pred[:,0])
    spearman = get_spearman(test_Y_cv, y_pred[:,0])
    rmse = get_rmse(test_Y_cv, y_pred[:,0])
    aupr = get_aupr(test_Y_cv, y_pred[:,0], threshold=12.1)
     
    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)
    print('ci:', ci2)
    print('AUPR', aupr)
    
 
    all_mse2[split] = mse
    all_r[split] = pearson
    all_aupr[split] = aupr
    all_rm2[split] = rm2
    all_ci2[split] = ci2

# In[10]:


print('cindex:',np.mean(all_ci), np.std(all_ci))
print('rm2:', np.mean(all_rm2), np.std(all_rm2))
print('mse:', np.mean(all_mse2), np.std(all_mse2))
print('pearson', np.mean(all_r), np.std(all_r))
print('AUPR', np.mean(all_aupr), np.std(all_aupr))
print('cindex:',np.mean(all_ci2), np.std(all_ci2))


# In[ ]:




