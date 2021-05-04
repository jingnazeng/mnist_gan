
$ docker run -it --rm tensorflow/tensorflow bash

Start a CPU-only container

$ docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python

Start a GPU container, using the Python interpreter.

$ docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-jupyter

Run a Jupyter notebook server with your own notebook directory (assumed here to be ~/notebooks). To use it, navigate to localhost:8888 in your browser.