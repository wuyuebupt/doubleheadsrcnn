from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import argparse
import pdb
import os
import scipy.io as sio
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
import os.path as osp
import sys

from PIL import Image

if __name__ == '__main__':
    root = sys.argv[1]
    annFile = sys.argv[2]
    annFile_basename = osp.basename(annFile)
    print (annFile_basename)

    output_folder = sys.argv[3]
    output = os.path.join(output_folder, annFile_basename + '.h5')

    root = os.path.expanduser(root)

    coco = COCO(annFile)
    ids = list(coco.imgs.keys())


    print (len(ids))
    ## read images
    with h5py.File(output, 'w') as f:
        # create group
        h5_group = f.create_group(annFile_basename)

        for index in ids:
            # print (index)
            # img_id = ids[index]
            img_id = index
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']

            imgpath = os.path.join(root, path)
            print (imgpath)
            img = Image.open(imgpath).convert('RGB')
            # print (img.size)

            key = path
            # print (key)
            # dset = h5_group.create_dataset(key, data=img, chunks=None)
            dset = h5_group.create_dataset(key, data=img)
            dset.attrs['path'] = key

            # exit()
