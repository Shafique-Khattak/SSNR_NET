
import os, sys
import matplotlib
import math
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, Conv1D, Conv2D, Embedding, LSTM
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, GlobalAveragePooling1D
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Reshape, Dropout, Dense,Multiply, Dot, concatenate, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.backend import squeeze
from tensorflow.keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import argparse
import pdb
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.losses import CategoricalCrossentropy, 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

random.seed(999)



epoch=10
batch_size=1

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
    
def make_category(class_info,len_data,num_classes):
    class_info = int(class_info)
    cats  = np.zeros((1,len_data,num_classes))
 
    for i in range(len_data):
        cats[0,i,class_info-1]=1      
   
    return cats
    
def train_data_generator(file_list_PS, noisy=False):
	index=0

	while True:
         PS_filepath=file_list_PS[index].split(',')
         signal, rate  = librosa.load(PS_filepath[2],sr=16000)
         signal=signal/np.max(abs(signal))
         F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)

         Lp=np.abs(F)
         phase=np.angle(F)
         if noisy==True:    
            meanR = np.mean(Lp, axis=1).reshape((257,1))
            stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
            NLp = (Lp-meanR)/stdR
         else:
            NLp=Lp
    
         noisy_LP = np.reshape(NLp.T,(1,NLp.shape[1],257))
     
         ssnr=np.asarray(float(PS_filepath[0])).reshape([1])
         final_len = noisy_LP.shape[1]
         frame_label = make_category(PS_filepath[1],final_len,10)
         
         label = make_category(PS_filepath[1],1,10)
         label =label.reshape(-1, label.shape[-1])
         index += 1       
         if index == len(file_list_PS):
             index = 0
            
             random.Random(7).shuffle(file_list_PS)

        
         yield  noisy_LP, [ssnr, ssnr[0]*np.ones([1,final_len,1]),label , frame_label]

def val_data_generator(file_list_PS, noisy=False):
	index=0

	while True:
         PS_filepath=file_list_PS[index].split(',')
         signal, rate  = librosa.load(PS_filepath[2],sr=16000)
         signal=signal/np.max(abs(signal))
         F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
            
         Lp=np.abs(F)
         phase=np.angle(F)
         if noisy==True:    
            meanR = np.mean(Lp, axis=1).reshape((257,1))
            stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
            NLp = (Lp-meanR)/stdR
         else:
            NLp=Lp
    
         noisy_LP = np.reshape(NLp.T,(1,NLp.shape[1],257))
     
         ssnr=np.asarray(float(PS_filepath[0])).reshape([1])
         final_len = noisy_LP.shape[1]
         frame_label = make_category(PS_filepath[1],final_len,10)
         
         label = make_category(PS_filepath[1],1,10)
         label =label.reshape(-1, label.shape[-1])
         index += 1       
         if index == len(file_list_PS):
             index = 0
            
             random.Random(7).shuffle(file_list_PS)

        
         yield  noisy_LP, [ssnr, ssnr[0]*np.ones([1,final_len,1]),label , frame_label]

def BLSTM_CNN_with_ATT_cross_domain():
    _input = Input(shape=(None, 257))
    re_input = keras.layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)
        
    # CNN
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
    conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
    re_shape = keras.layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
 
    blstm=Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(re_shape)

    flatten = TimeDistributed(keras.layers.Flatten())(blstm)
    dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1=Dropout(0.3)(dense1)
     
    frame_label = TimeDistributed(Dense(10, activation='softmax'),name='frame_label')(flatten)
    label = GlobalAveragePooling1D(name = 'label')(frame_label)
    
    merge_input = concatenate([dense1, frame_label])     
    
    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention')(merge_input)
    Frame_score=TimeDistributed(Dense(1), name='Frame_score')(attention)
    ssnr_score=GlobalAveragePooling1D(name='ssnr_score')(Frame_score)
    model = Model(outputs=[ssnr_score, Frame_score, label, frame_label], inputs=_input)
    
    return model
    

print ('model building...')
    
model = BLSTM_CNN_with_ATT_cross_domain()
pathmodel="TIMIT_SSNR_PS"    

alpha = 0.6  
adam = Adam(lr=1e-4)
model.compile(loss={'ssnr_score': 'mse', 'Frame_score': 'mse', 'label':'categorical_crossentropy','frame_label':'categorical_crossentropy'},loss_weights={'ssnr_score': alpha, 'label': 1-alpha}, optimizer=adam, metrics={'label':'accuracy','frame_label':'accuracy'},)
plot_model(model, to_file='model_'+pathmodel+'.png', show_shapes=True)

with open(pathmodel+'.json','w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath=pathmodel+'.hdf5', verbose=1, save_best_only=True, mode='min')  

print ('training...')
 

Train_list_PS = ListRead('Train_List.txt')
Num_train =  len(Train_list_PS)
#Train_List.txt example
'''
ssnr,class_lable,filepath
15.314,1,/Data/user_ahmed/TIMIT_SSNR/Data/TIMIT_SSNR_V1/train/Noisy/cafeteria_babble/1.wav
13.681,1,/Data/user_ahmed/TIMIT_SSNR/Data/TIMIT_SSNR_V1/train/Noisy/cafeteria_babble/10.wav
''' 
Test_list_PS = ListRead('Test_List.txt')
Num_testdata= len (Test_list_PS)

    
g1 = train_data_generator(Train_list_PS, noisy=False)
g2 = val_data_generator  (Test_list_PS, noisy=False)

hist=model.fit(g1,steps_per_epoch=Num_train, epochs=epoch, verbose=1,validation_data=g2,validation_steps=Num_testdata,max_queue_size=1, workers=1,callbacks=[checkpointer])

pathmodel="TIMIT_SSNR_PS"
model.save(pathmodel+'.h5')

# plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
print ('drawing the training process...')
plt.figure(2)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('Learning_curve_TIMIT_SSNR_PS.png', dpi=150)


print ('load model...')
    
model = BLSTM_CNN_with_ATT_cross_domain()
model.load_weights(pathmodel+'.hdf5')
    
print ('testing...')
SSNR_Predict=np.zeros([len(Test_list_PS),])
SSNR_true   =np.zeros([len(Test_list_PS),])
label_Predict =[]
label_true =[]
    
for i in range(len(Test_list_PS)):
    PS_filepath=Test_list_PS[i].split(',')
    noisy_LP =np.load(PS_filepath[2])       
    ssnr=float(PS_filepath[0])

    [SSNR_score, frame_score, label, frame_label]=model.predict(noisy_LP, verbose=0, batch_size=batch_size)

    SSNR_Predict[i] = SSNR_score
    SSNR_true[i] = ssnr
        
    label_=make_category(PS_filepath[1],1,10)
    label_ =label_.reshape(-1, label_.shape[-1])
        
    label_Predict.append(label)
    label_true.append(label_)

MSE=np.mean((SSNR_true-SSNR_Predict)**2)
print ('Test error= %f' % MSE)
LCC=np.corrcoef(SSNR_true, SSNR_Predict)
print ('Linear correlation coefficient= %f' % LCC[0][1])
SRCC=scipy.stats.spearmanr(SSNR_true.T, SSNR_Predict.T)
print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot SSNR
M=np.max([np.max(SSNR_Predict),30])
plt.figure(1)
plt.scatter(SSNR_true, SSNR_Predict, s=30)
plt.xlim([0,M])
plt.ylim([0,M])
plt.xlabel('True SSNR')
plt.ylabel('Predicted SSNR')
plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
plt.show()
plt.savefig('Scatter_plot_TIMIT_SSNR_PS.png', dpi=150)
    
label_Predict = np.array(label_Predict)
label_Predict =label_Predict.reshape(-1, label_Predict.shape[-1])
label_Predict = np.argmax(label_Predict, axis=1)
label_true = np.array(label_true)
label_true =label_true.reshape(-1, label_true.shape[-1])
label_true = np.argmax(label_true, axis=1)

report = classification_report(label_true, label_Predict)
report_path = "report_TIMIT_SSNR_PS.txt"

text_file = open(report_path, "w")
text_file.write(report)
text_file.close()
