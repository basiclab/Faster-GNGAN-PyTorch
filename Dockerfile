FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN pip3 install \
    absl-py \
    lmdb \
    tqdm \
    pytorch-gan-metrics \
    tensorboardX

RUN mkdir /gngan
WORKDIR /gngan
