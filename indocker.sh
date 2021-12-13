#! /bin/bash

podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
	--ipc=host \
	-v ${PWD}:/gngan \
	-it \
	-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
	w86763777:ngc-21.11-gngan \
	$@
