FROM nvcr.io/nvidia/pytorch:21.11-py3

RUN pip3 install absl-py \
    h5py \
    lmdb \
    tqdm \
    scipy==1.5.4 \
    pytorch-gan-metrics \
    tensorboardX

RUN mkdir /gngan
WORKDIR /gngan
