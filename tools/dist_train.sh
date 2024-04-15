#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
python -m torch.distributed.launch --nproc_per_node=${NGPUS} train_track.py --launcher pytorch ${PY_ARGS}


