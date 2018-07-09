export CUDA_VISIBLE_DEVICES=0,1,2,3
METHOD=$1
DATASET=$2
MODEL_NAME=$3
SPLIT=$4
NUM_GPU=$5
sh clean.sh $DATASET $METHOD $SPLIT
python tools/test.py \
models/$DATASET/$METHOD/${SPLIT}/deploy_tpl.prototxt \
models/$DATASET/$METHOD/${SPLIT}/proto_splits/ \
models/$DATASET/$METHOD/${SPLIT}/model/${MODEL_NAME}.caffemodel \
data/${DATASET}_flow_val_split_${SPLIT}.txt \
--num_worker \
$NUM_GPU \
--save_scores \
final_scores/${DATASET}_${MODEL_NAME}_split${SPLIT}
