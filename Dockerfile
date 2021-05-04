# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

FROM tensorflow/tensorflow:2.3.0-gpu

RUN pip install Flask==1.1.2
RUN pip install Pillow==8.1.0
RUN pip install matplotlib==3.3.3

COPY . /app

WORKDIR /app

