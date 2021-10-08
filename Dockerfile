FROM nvcr.io/nvidia/pytorch:21.07-py3

RUN pip3 install absl-py \
    h5py \
    lmdb \
    tqdm \
    scipy==1.5.4 \
    pytorch-gan-metrics==0.3.2 \
    tensorboardX

RUN mkdir /gngan
WORKDIR /gngan
