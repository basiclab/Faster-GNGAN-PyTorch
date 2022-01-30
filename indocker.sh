#! /bin/bash

if [ -z "${CUDA_VISIBLE_DEVICES// }" ]; then
    gpuids=0
else
    gpuids="$CUDA_VISIBLE_DEVICES"
fi

podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
	--ipc=host \
	-v ${PWD}:/gngan \
	-v $(realpath data):$(realpath data) \
	-v $(realpath logs):$(realpath logs) \
	-v $(realpath stats):$(realpath stats) \
	-it \
	-e CUDA_VISIBLE_DEVICES=$gpuids \
	w86763777:ngc-21.11-gngan \
	$@
