from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys
import time
import ijson

# import _init_paths
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

from models.pose_hrnet import get_pose_net

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def main():
    
    ## python inference.py --cfg ../experiments/atrw/w48_384x288.yaml TEST.MODEL_FILE ../../output/atrw/pose_hrnet/w48_384x288/final_state.pth
    path_to_images =                '/root/hrnet/code/deep-high-resolution-net.pytorch-master/data/atrw/images/val/'
    path_to_annotation_file =       '/root/hrnet/code/deep-high-resolution-net.pytorch-master/data/atrw/annotations/keypoint_val.json'
    path_to_keypoint_error_file =   '/root/hrnet/code/deep-high-resolution-net.pytorch-master/demo/output_atrw/error_per_image_per_keypoint.txt'
    path_to_output =                '/root/hrnet/code/deep-high-resolution-net.pytorch-master/demo/output_atrw/'
    n_KP = 15

    num_images_to_test = 100
    
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    pose_model = eval('get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    f2 = open(path_to_annotation_file)
    images = ijson.items(f2, 'images.item')    
    count = num_images_to_test

    MSE_KP = np.zeros(n_KP)

    output_file = open(path_to_keypoint_error_file, 'w')
    
    kp_str = ["KP_%s" % i for i in range(1, 28)]
    output_file.write("FILENAME," + ','.join(kp_str) + "\n")

    for obj in images:
        input_img_name = obj['filename']
        print(input_img_name)
        input_img = path_to_images + input_img_name
        output_img = path_to_output + input_img_name
        print(output_img)

        f1 = open(path_to_annotation_file)
        annotations = ijson.items(f1, 'annotations.item')
        annotations = ([[(int(o['bbox'][0]), int(o['bbox'][1])), (int(o['bbox'][0]) + int(o['bbox'][2]), int(o['bbox'][1]) + int(o['bbox'][3]))], o['keypoints']] for o in annotations if o['image_id'] == obj['id'])
        gt_bboxes = []
        gt_keypoints = []
        for i in annotations:
            gt_bboxes.append(i[0])
            temp = np.reshape([float(k) for k in i[1]], (n_KP, 3))
            gt_keypoints.append(np.delete(temp, 2, 1))
        f1.close()

        image_bgr = cv2.imread(input_img)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for detection and pose estimation
        if cfg.DATASET.COLOR_RGB:
            image_per = image_rgb.copy()
            image_pose = image_rgb.copy()
        else:
            image_per = image_bgr.copy()
            image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        for box in gt_bboxes:
            cv2.rectangle(image_debug, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), color=(0, 255, 0), thickness=3)

        centers = []
        scales = []
        for box in gt_bboxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        now = time.time()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        
        for i in range(len(pose_preds)):
            A = pose_preds[i]
            B = gt_keypoints[i]
            mse = (np.square(A - B)).mean(axis=1)
            MSE_KP = MSE_KP + mse
            mse = ["%.2f" % i for i in mse]
            output_file.write(input_img_name + "," + ','.join(mse) + "\n")
            
        new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])

        total_then = time.time()

        text = "{:03.2f} sec".format(then - now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(output_img, image_debug)

        count -= 1
        if (count == 0):
            break
    
    total_error = MSE_KP/num_images_to_test
    total_error_str = ["%.2f" % i for i in total_error]
    output_file.write("TOTAL ERROR" + "," + ','.join(total_error_str) + "\n")
    output_file.close()
    f2.close()

    return 0


if __name__ == '__main__':
    main()