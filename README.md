# ObjectDetectionWebcam_MXNet

Simple script to run object detection from a webcam using opencv and matplotlib, using a [gluon-cv](https://gluon-cv.mxnet.io/model_zoo/detection.html) SSD pretrained model with a mobilenet backend.

usage:

```bash
pythonw detection.py
```

Note, on MacOS, you need to install python as a framework, see this [page](https://matplotlib.org/faq/osx_framework.html)

for conda environments, simply run:
```bash
conda install python.app
```

Result:

![](https://media.giphy.com/media/9JvoKeUeCt4bdRf3Cv/giphy.gif)
