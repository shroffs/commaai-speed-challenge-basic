#!/usr/bin/env python3

import tensorflow as tf
import cv2
import numpy as np
import functools 
from tqdm import tqdm

tf.keras.backend.clear_session()

# create a funciton that converts a video to a numpy array of frames
def video2numpy(vid):
    vid_cap = cv2.VideoCapture(vid)
    num = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    vid_array = np.empty((num, 480, 640, 3), dtype=np.uint8)

    frame_num = 0
    ret = True

    while (frame_num < num and ret):
        ret , frame = vid_cap.read()
        vid_array[frame_num] = frame
        frame_num +=1

    vid_cap.release()

    return vid_array

# create a function that turn line separated data into an array

def lsv2numpy(lsv):
    txt = open(lsv, 'r')
    values = [float(line.rstrip('\n')) for line in txt]
    return np.array(values)

#create a function that gets a batch of sequences from the data

def get_batch(X, y_true, seq_length, batch_size):
    #return batch_size number of sequences seq_length long
    
    #find highest index of data
    max_idx = X.shape[0]-1
    #randomly select batch_size number of indexes
    idx = np.random.choice(max_idx-seq_length, batch_size)
    
    #get sequences from data
    input_batch = []
    output_batch = []
    for i in idx:
        input_batch.append(X[i:i+seq_length])
        output_batch.append(y_true[i:i+seq_length])

    return np.array(input_batch, dtype=np.float32), np.array(output_batch)


def LSTM(filters):
    return tf.keras.layers.ConvLSTM2D(
            filters,
            (3,3),
            strides=(3,3),
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            recurrent_activation='sigmoid',
            stateful=True)

def build_model():
    model = tf.keras.Sequential([
        #put through LSTM
        LSTM(90),
        LSTM(40),
        LSTM(10),
        #out one value that is predicted speed
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
        ])

    return model

#create function that computes MSE loss

def compute_loss(y_pred, y_true):
    loss = tf.keras.losses.logcosh(y_true, y_pred)
    return loss

#optimizer parameters:
training_iter = 30000
batch_size = 5
fps = 20
seq_length = int(fps * 0.25)
learning_rate = 1e-3

#Initialize things
X = video2numpy('/home/spencer/speedchallenge/data/train.mp4')
y_true = lsv2numpy('/home/spencer/speedchallenge/data/train.txt')

optimizer = tf.keras.optimizers.Adadelta()
model = build_model()

@tf.function
def train_step(x, y_true):
    with tf.GradientTape() as tape:
        #make forward pass
        y_pred = model(x)
        #compute loss
        loss = compute_loss(y_pred, y_true)

    #compute grads
    grads = tape.gradient(loss, model.trainable_variables)
    #take training step
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

val_loss = 0
for iter in tqdm(range(training_iter)):
    #get a batch of input and values as well as validation data
    x_train, y_train = get_batch(X, y_true, batch_size, seq_length)
    #take a training step
    train_loss = train_step(x_train, y_train)

    if iter % 12:
        x_val, y_val = get_batch(X, y_true, batch_size, seq_length)
        val_loss += compute_loss(model(x_val), y_val).numpy().mean()

    if iter % 50:
        print(train_loss.numpy().mean())
        print(val_loss)
        val_loss = 0

model.summary()

X = []
y_true = []
X_test = video2numpy("/home/spencer/speedchallenge/data/test.mp4")

def predict_speeds(X, seq_length):
    #return batch_size number of sequences seq_length long

    #randomly select batch_size number of indexes
    idx = range(0, X.shape[0]-seq_length)

    #get sequences from data
    speeds = []
    for i in idx:
        x = np.expand_dims(np.array(X[i:i+seq_length], dtype=np.float32),0)
        speeds.append(model.predict(x))

    return np.array(speeds, dtype=np.float32)

speeds = predict_speeds(X_test, 5)
np.save('test2.npy', speeds)



