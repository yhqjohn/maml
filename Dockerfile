FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install torchvision\
    && pip install metann\
    && rm -rf ~/.cache/pip

ENV GLOO_SOCKET_IFNAME=eth0

WORKDIR /work
COPY data ./data
COPY utils ./utils
COPY vision ./vision
COPY maml.py models.py MiniImagenet.py models.py remote_train.py ./