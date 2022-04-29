#! /bin/bash

if [ -z "${CUDA_VISIBLE_DEVICES// }" ]; then
    gpuids=0
else
    gpuids="$CUDA_VISIBLE_DEVICES"
fi

docker run \
	--ipc=host \
    --gpus all \
	-v ${PWD}:/gngan \
	-v $(realpath data):$(realpath data) \
	-v $(realpath logs):$(realpath logs) \
	-v $(realpath stats):$(realpath stats) \
	-v $(realpath ~/.cache/torch/hub):/root/.cache/torch/hub \
	-it \
	-e CUDA_VISIBLE_DEVICES=$gpuids \
	yilun:ngc-22.02-gngan \
	$@

# podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
# 	--ipc=host \
# 	-v ${PWD}:/gngan \
# 	-v $(realpath data):$(realpath data) \
# 	-v $(realpath logs):$(realpath logs) \
# 	-v $(realpath stats):$(realpath stats) \
# 	-v $(realpath ~/.cache/torch/hub):/root/.cache/torch/hub \
# 	-it \
# 	-e CUDA_VISIBLE_DEVICES=$gpuids \
# 	w86763777:ngc-22.02-gngan \
# 	$@