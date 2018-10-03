import time

import cv2
import gluoncv as gcv
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

# Load the model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

# Load the webcam handler
cap = cv2.VideoCapture(0)
axes = None
time.sleep(1) ### letting the camera autofocus

while True:
    # Load frame from the camera
    ret, frame = cap.read()
    frame = cv2.resize(frame, (700, 512))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_nd = mx.nd.array(rgb, dtype='float32').transpose((2, 0, 1))/255.
    rgb_nd = mx.nd.image.normalize(rgb_nd, mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)).expand_dims(axis=0)

    # Run frame through network
    class_IDs, scores, bounding_boxes = net(rgb_nd)
  
    # Display the result
    plt.cla()
    axes = gcv.utils.viz.plot_bbox(rgb, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, ax=axes)
    plt.draw()
    plt.pause(0.001)

# When everything is done, release the capture
cap.release()
