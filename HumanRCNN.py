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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_session(gpu_fraction):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess = get_session(0.5)

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
n_epochs= 100
batch_size1 = 768

# ds_file, label_index, rst_file, use_emb, hiddem_dim
ds_file1 = '../../Dataset/ConfidenceScore/train50.txt'
ds_file2 = '../../Dataset/ConfidenceScore/test50.txt'
label_index = 2
rst_file = 'results/yeast_rcnn2.txt'
sid1_index = 0
sid2_index = 1
# if len(sys.argv) > 1:
#     ds_file, label_index, rst_file, use_emb, hidden_dim, n_epochs = sys.argv[1:]
#     label_index = int(label_index)
#     use_emb = int(use_emb)
#     hidden_dim = int(hidden_dim)
#     n_epochs = int(n_epochs)

seq2t = s2t(emb_files[use_emb])

max_data = -1
limit_data = max_data > 0
raw_data = []
raw_data2 = []
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


def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPooling1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPooling1D(3)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPooling1D(3)(l5(s1))
    s1=concatenate([r5(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPooling1D(3)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPooling1D(3)(l5(s2))
    s2=concatenate([r5(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model


adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-5)
rms = RMSprop(lr=0.001)

# from sklearn.model_selection import KFold, ShuffleSplit
# kf = KFold(n_splits=10, shuffle=True)
# tries = 3
# cur = 0
# recalls = []
# accuracy = []
# total = []
# total_truth = []
# train_test = []
# for train, test in kf.split(class_labels):
#     # if np.sum(class_labels[train], 0)[0] > 0.8 * len(train) or np.sum(class_labels[train], 0)[0] < 0.2 * len(train):
#     #     continue
#     train_test.append((train, test))
#     cur += 1
#     if cur >= tries:
#         break


from keras.models import load_model
from keras.utils import multi_gpu_model
#train, test = train_test[0]

for j in range(19):

    # copy below
    num_hit = 0.
    num_total = 0.
    num_pos = 0.
    num_true_pos = 0.
    num_false_pos = 0.
    num_true_neg = 0.
    num_false_neg = 0.

    class_labels = np.zeros((len(raw_data), 2))
    n_classifier = j
    threshold = n_classifier * 0.05 + 0.05
    for i in range(len(raw_data)):
        flag = float(raw_data[i][label_index]) < threshold
        flag = int(flag)
        class_labels[i][flag] = 1.
    class_labels2 = np.zeros((len(raw_data2), 2))
    for i in range(len(raw_data2)):
        flag = float(raw_data2[i][label_index]) < threshold
        flag = int(flag)
        class_labels2[i][flag] = 1.

    merge_model = None
    merge_model = build_model()
    #parallel_model = multi_gpu_model(merge_model, gpus=3)

    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)

    #parallel_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #parallel_model.fit([seq_tensor[seq_index1], seq_tensor[seq_index2]], class_labels, batch_size=batch_size1, epochs=n_epochs)

    merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    merge_model.fit([seq_tensor[seq_index1], seq_tensor[seq_index2]], class_labels, batch_size=batch_size1,
                    epochs=100)

    merge_model.save('ClassifiedModel/Model' + str(j + 1) + 'epoch' + str(100) + 'batchsize' + str(batch_size1) + '.h5')
    # model = load_model('ClassifiedModel/Model' + str(j + 1) + 'epoch' + str(60 + 10 * k) + '.h5')

    pred = merge_model.predict([seq_tensor[seq_index3], seq_tensor[seq_index4]])
    for i in range(len(class_labels2)):
        num_total += 1
        if np.argmax(class_labels2[i]) == np.argmax(pred[i]):
            num_hit += 1
        if class_labels2[i][0] > 0.:
            num_pos += 1.
            if pred[i][0] > pred[i][1]:
                num_true_pos += 1
            else:
                num_false_neg += 1
        else:
            if pred[i][0] > pred[i][1]:
                num_false_pos += 1
            else:
                num_true_neg += 1
    accuracy = num_hit / num_total
    prec = num_true_pos / (num_true_pos + num_false_pos)
    recall = num_true_pos / num_pos
    spec = num_true_neg / (num_true_neg + num_false_neg)
    f1 = 2. * prec * recall / (prec + recall)

    print(accuracy, prec, recall, spec, f1)

    # file = open('results/model' + str(j + 1) + '.txt','a')
    # for token in pred:
    #     file.write(str(np.argmax(token)) + '\n')
    # file.close()
    with open(rst_file, 'a') as fp:
        fp.write(str(j) + ' ' + id2seq_file + ' ' + str(batch_size1) + ' ' + str(hidden_dim) + ' ' + 'epoch = ' + str(
            n_epochs) + ' ' + 'acc=' + str(accuracy) + '\tprec=' + str(prec) + '\trecall=' + str(
            recall) + '\tspec=' + str(spec) + '\tf1=' + str(f1) + '\n')