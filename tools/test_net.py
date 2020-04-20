# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import ast

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    ## add for training
    parser.add_argument(
        "--data-dir",
        default="",
        metavar="DIR",
        help="path to data folder",
        type=str,
    )
    parser.add_argument(
        "--pretrained-model",
        default="",
        help="path to pretrained model",
        metavar="FILE",
        type=str,
    )
    parser.add_argument(
        "--nonlocal-cls-num-group",
        default="1",
        help="nonlocal num group cls",
        metavar="1",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-cls-num-stack",
        default="0",
        help="nonlocal num stack cls",
        metavar="1",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-reg-num-group",
        default="1",
        help="nonlocal num group reg",
        metavar="1",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-reg-num-stack",
        default="0",
        help="nonlocal num stack reg",
        metavar="1",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-shared-num-group",
        default="1",
        help="nonlocal num group reg",
        metavar="1",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-shared-num-stack",
        default="0",
        help="nonlocal num stack reg",
        metavar="1",
        type=int,
    )

    parser.add_argument(
        "--nonlocal-out-channels",
        default="2048",
        help="nonlocal out channels for fpn, fpn=2048(like c4)",
        metavar="2048",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-inter-channels",
        default="256",
        help="nonlocal inter channels, c4 < 2048, fpn < 256",
        metavar="256",
        type=int,
    )
    parser.add_argument(
        "--nonlocal-use-shared",
        default="True",
        help="nonlocal use shared non-locael",
        metavar="True",
        type=str,
    )
    parser.add_argument(
        "--nonlocal-use-bn",
        default="True",
        help="nonlocal use bn after attention",
        metavar="True",
        type=str,
    )
    parser.add_argument(
        "--nonlocal-use-softmax",
        default="False",
        help="nonlocal use softmax other than div",
        metavar="False",
        type=str,
    )
    parser.add_argument(
        "--nonlocal-use-attention",
        default="True",
        help="nonlocal use attention before ffconv",
        metavar="True",
        type=str,
    )

    parser.add_argument(
        "--nonlocal-use-ffconv",
        default="True",
        help="nonlocal use ffconv after nonlocal with residual",
        metavar="True",
        type=str,
    )
    parser.add_argument(
        "--nonlocal-use-relu",
        default="True",
        help="nonlocal use relu after bn",
        metavar="True",
        type=str,
    )
    parser.add_argument(
        "--conv-bbox-expand",
        default="1.0",
        help="box expand conv",
        metavar="1.0",
        type=float,
    )
    parser.add_argument(
        "--fc-bbox-expand",
        default="1.0",
        help="box expand fc",
        metavar="1.0",
        type=float,
    )
    parser.add_argument(
        "--backbone-out-channels",
        default="256",
        help="fpn out channels for fpn, fpn=2048(like c4)",
        metavar="256",
        type=int,
    )

    parser.add_argument(
        "--evaluation-flags",
        nargs = '*',
        # default=[0, 3],
        default=[],
        help="model code for evaluation flags",
        metavar="1 1 1 1",
        type=int,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    cfg.DATA_DIR = args.data_dir
    cfg.MODEL.WEIGHT = args.pretrained_model
    cfg.MODEL.ROI_BOX_HEAD.NEIGHBOR_CONV_EXPAND = args.conv_bbox_expand
    cfg.MODEL.ROI_BOX_HEAD.NEIGHBOR_FC_EXPAND = args.fc_bbox_expand

    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_CLS_NUM_GROUP = args.nonlocal_cls_num_group
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_CLS_NUM_STACK = args.nonlocal_cls_num_stack
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_REG_NUM_GROUP = args.nonlocal_reg_num_group
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_REG_NUM_STACK = args.nonlocal_reg_num_stack
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_SHARED_NUM_GROUP = args.nonlocal_shared_num_group
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_SHARED_NUM_STACK = args.nonlocal_shared_num_stack
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_INTER_CHANNELS = args.nonlocal_inter_channels
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_OUT_CHANNELS = args.nonlocal_out_channels

    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_SHARED = ast.literal_eval(args.nonlocal_use_shared)
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_BN = ast.literal_eval(args.nonlocal_use_bn)
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_SOFTMAX = ast.literal_eval(args.nonlocal_use_softmax)
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_FFCONV = ast.literal_eval(args.nonlocal_use_ffconv)
    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_RELU = ast.literal_eval(args.nonlocal_use_relu)

    cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_ATTENTION = ast.literal_eval(args.nonlocal_use_attention)

    cfg.MODEL.BACKBONE.OUT_CHANNELS = args.backbone_out_channels

    # double heads
    cfg.TEST.EVALUATION_FLAGS = args.evaluation_flags


    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
