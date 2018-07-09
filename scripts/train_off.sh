#!/usr/bin/env bash
DATASET=$1
MODALITY=$2
SPLIT=$3
TOOLS=lib/caffe-action/build/install/bin
N_GPU=4
MPI_BIN_DIR=/usr/local/openmpi-1.8.5/bin/
NOW="`date +%Y%m%d%H%M%S`"
LOG_FILE=logs/${NOW}_${DATASET}_${MODALITY}_split${SPLIT}.log
echo "logging to ${LOG_FILE}"

${MPI_BIN_DIR}mpirun -np $N_GPU \
$TOOLS/caffe train --solver=models/${DATASET}/${MODALITY}/${SPLIT}/solver.prototxt  \
--weights=models/bn_inception_${DATASET}_${MODALITY}_${SPLIT}_init.caffemodel 2>&1 | tee ${LOG_FILE}
