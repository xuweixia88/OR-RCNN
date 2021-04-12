from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../WordEmbedding' not in sys.path:
    sys.path.append('../WordEmbedding')

from ORRCNN.WordEmbedding.seq2tensor import s2t
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, merge
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam,  RMSprop

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def get_session(gpu_fraction=0.9):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess = get_session(0.9)

KTF.set_session(sess)

import numpy as np
from tqdm import tqdm

from keras.layers import Input, CuDNNGRU
from numpy import linalg as LA
import scipy

# change
id2seq_file =  '../../Dataset/ConfidenceScore/Sequence.txt'

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split()
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
seq_array = []
id2_aid = {}
sid = 0

seq_size = 2000
emb_files = ['../../../embeddings/default_onehot.txt', '../WordEmbedding/string_vec5.txt', '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
use_emb = 1
hidden_dim = 50
n_epochs= 150

# ds_file, label_index, rst_file, use_emb, hiddem_dim
ds_file1 = '../../Dataset/ConfidenceScore/train.txt'
ds_file2 = '../../Dataset/ConfidenceScore/test.txt'
label_index = 2
rst_file = 'results/rcnn1.txt'
sid1_index = 0
sid2_index = 1
use_log = 0
# if len(sys.argv) > 1:
#     ds_file, label_index, rst_file, use_emb, hidden_dim, n_epochs, use_log = sys.argv[1:]
#     label_index = int(label_index)
#     use_emb = int(use_emb)
#     hidden_dim = int(hidden_dim)
#     n_epochs = int(n_epochs)

seq2t = s2t(emb_files[use_emb])

max_data = -1
limit_data = max_data > 0
raw_data = []
raw_data2 = []
raw_ids = []
skip_head = False
x = None
count = 0

for line in tqdm(open(ds_file1)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split(' ')
    if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
        continue
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]
    raw_data.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break
print (len(raw_data))

for line in tqdm(open(ds_file2)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split(' ')
    if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
        continue
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]
    raw_data2.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break

len_m_seq = np.array([len(line.split()) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)
print (avg_m_seq, max_m_seq)

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])
seq_index3 = np.array([line[sid1_index] for line in tqdm(raw_data2)])
seq_index4 = np.array([line[sid2_index] for line in tqdm(raw_data2)])

print(seq_index1[:10])

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1 = Conv1D(hidden_dim, 3)
    r1 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2 = Conv1D(hidden_dim, 3)
    r2 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3 = Conv1D(hidden_dim, 3)
    r3 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4 = Conv1D(hidden_dim, 3)
    r4 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5 = Conv1D(hidden_dim, 3)
    r5 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6 = Conv1D(hidden_dim, 3)
    s1 = MaxPooling1D(2)(l1(seq_input1))
    s1 = concatenate([r1(s1), s1])
    s1 = MaxPooling1D(2)(l2(s1))
    s1 = concatenate([r2(s1), s1])
    s1 = MaxPooling1D(3)(l3(s1))
    s1 = concatenate([r3(s1), s1])
    s1 = l6(s1)
    s1 = GlobalAveragePooling1D()(s1)
    s2 = MaxPooling1D(2)(l1(seq_input2))
    s2 = concatenate([r1(s2), s2])
    s2 = MaxPooling1D(2)(l2(s2))
    s2 = concatenate([r2(s2), s2])
    s2 = MaxPooling1D(3)(l3(s2))
    s2 = concatenate([r3(s2), s2])
    s2 = l6(s2)
    s2 = GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(1, activation='sigmoid')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

batch_size1 = 128
adam = Adam(lr=0.005, amsgrad=True, epsilon=1e-5)



num_total = 0.
total_mse = 0.
total_mae = 0.
total_cov = 0.


fp2 = open('temp.txt', 'w')
for j in [0]:
    merge_model = build_model()
    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)

    score_labels = np.zeros((len(raw_data), 1))
    for i in range(len(raw_data)):
        score_labels[i] = float(raw_data[i][label_index])
    score_labels2 = np.zeros((len(raw_data2), 1))
    for i in range(len(raw_data2)):
        score_labels2[i] = float(raw_data2[i][label_index])
    merge_model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
    merge_model.fit([seq_tensor[seq_index1], seq_tensor[seq_index2]], score_labels, batch_size=batch_size1, epochs=n_epochs)
    #result1 = merge_model.evaluate([seq_tensor1[test], seq_tensor2[test]], score_labels[test])
    merge_model.save('regression.h5')
    pred = merge_model.predict([seq_tensor[seq_index3], seq_tensor[seq_index4]])

    this_mae, this_mse, this_cov = 0., 0., 0.
    this_num_total = 0
    for i in range(len(score_labels2)):
        this_num_total += 1
        diff = abs(score_labels2[i] - pred[i])
        this_mae += diff
        this_mse += diff**2

    num_total += this_num_total
    total_mae += this_mae
    total_mse += this_mse
    mse = total_mse / num_total
    mae = total_mae / num_total
    this_cov = scipy.stats.pearsonr(np.ndarray.flatten(pred), score_labels2)[0]
    # for i in range(len(raw_data2)):
    #     fp2.write(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(np.ndarray.flatten(pred)[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(np.ndarray.flatten(pred)[i]) + '\n')
    # print(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(np.ndarray.flatten(pred)[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(np.ndarray.flatten(pred)[i]))
    total_cov += this_cov
    print (mse, mae, this_cov)
fp2.close()

# mse = total_mse / num_total
# mae = total_mae / num_total
# total_cov /= len(train_test)
# print (mse, mae, total_cov)

# with open(rst_file, 'w') as fp:
#     fp.write('mae=' + str(mae) + '\nmse=' + str(mse) + '\ncorr=' + str(total_cov))