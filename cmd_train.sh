export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=3
### path to save the models 
export OUTPUT_DIR=/path/to/modelfolder/
### path to data folder
export DATA_DIR=/path/to/datafolder/

### training using 4 gpus with a batchsize 8 (2 images per gpu)
### Resnet 50, FPN
export CONFIG_YAML=configs/double_heads/e2e_dh_faster_rcnn_R_50_FPN_1x_bs8.yaml
export PRETRAIN_MODEL=/path/to/pretrained/model
export OUT_CHANNELS=256
export NONLOCAL_OUT_CHANNELS=1024
export INTER_CHANNELS=512

### Resnet 101, FPN
# export CONFIG_YAML=configs/double_heads/e2e_dh_faster_rcnn_R_101_FPN_1x_bs8.yaml
# export PRETRAIN_MODEL=/path/to/pretrained/model
# export OUT_CHANNELS=256
# export NONLOCAL_OUT_CHANNELS=1024
# export INTER_CHANNELS=512


python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--output-dir $OUTPUT_DIR  \
--pretrained-model $PRETRAIN_MODEL \
--data-dir $DATA_DIR \
--config-file $CONFIG_YAML \
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
--mask-loss 0.5 2.0 1.4 0.6 \
--evaluation-flags 1 1 1 1 \
--lr-steps 120000 160000 180000

####### lr schedule: with batch size 8, init lr 0.01 ##################
# 1x, 180k: decrease at 120000 and 160000, end at 180000, This schedules results in 12.17 epochs over the 118,287 images in coco_2017_train (or equivalently, coco_2014_train union coco_2014_valminusminival). 
# --lr-steps 120000 160000 180000
# 
# 2x: twice
# --lr-steps 240000 320000 360000
#
# s1x: stretched 1x, 1.44x + extends the duration of the first learning rate
# --lr-steps 200000 240000 260000 
#
# 280k: cascade RCNN, (the setting used: FPN+(RoIAlign) , ResNet101, 512 ROIs, batchsize 8, init lr 0.005)
# --lr-steps 160000 240000 280000
#######################################################################

