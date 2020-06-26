# commaai-speed-challenge-basic
This is a bare-bones network that is trained on 20fps raw video from a dashcam to predict speed (in mph). The model is a Convolutional LSTM that takes an input sequence of video frames and outputs speed. 
No preprocessing of the video data was done before hand to keep this model full generalizable to other video datasets. For performance, extensive preprocessing of frames and data augmentations should be used.
This repo is more a guide for basic Tensorflow setup for LSTM and video as well as a platform for experimenting more with in the future (optical flow, etc).
