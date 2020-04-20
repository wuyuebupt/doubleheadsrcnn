export PYTHONPATH=$PWD/maskrcnn_pythonpath

# you may use multiple gpus, remember to modify the TEST.IMS_PER_BATCH in test.yaml
export NGPUS=1
export DATA_DIR=/path/to/datafolder/

### Resnet 50, FPN
# export MODEL_PATH=./models/dh_faster_rcnn_R_50_FPN_1x.pth
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_dh_faster_rcnn_R_50_FPN_1x_test.yaml
# export OUT_CHANNELS=256
# export NONLOCAL_OUT_CHANNELS=1024
# export INTER_CHANNELS=512

### Resnet 101, FPN
export MODEL_PATH=./models/dh_faster_rcnn_R_101_FPN_1x.pth
export CONFIG_YAML=configs/double_heads/e2e_dh_faster_rcnn_R_101_FPN_1x_test.yaml
export OUT_CHANNELS=256
export NONLOCAL_OUT_CHANNELS=1024
export INTER_CHANNELS=512


python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
--config-file $CONFIG_YAML \
--pretrained-model $MODEL_PATH \
--data-dir $DATA_DIR \
--nonlocal-cls-num-group 4 \
--nonlocal-cls-num-stack 0 \
--nonlocal-reg-num-group 4 \
--nonlocal-reg-num-stack 0 \
--nonlocal-shared-num-group 4 \
--nonlocal-shared-num-stack 2 \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax False \
--nonlocal-use-ffconv True \
--nonlocal-use-attention True \
--nonlocal-inter-channels $INTER_CHANNELS \
--nonlocal-out-channels $NONLOCAL_OUT_CHANNELS \
--conv-bbox-expand  1.3 \
--fc-bbox-expand  1.0 \
--backbone-out-channels $OUT_CHANNELS \
--evaluation-flags 1 1 1 1 \
