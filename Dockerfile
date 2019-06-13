FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install torchvision \
    && rm -rf ~/.cache/pip

ENV GLOO_SOCKET_IFNAME=eth0

WORKDIR /work
COPY maml.py meta.py MiniImagenet_old.py models.py remote_train.py ./
COPY miniimagenet ./miniimagenet