import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask




def dice_score(result_mask, ground_truth_mask):
    result_mask = np.array(result_mask) > 0
    ground_truth_mask = np.array(ground_truth_mask) > 0

    intersection = np.sum(result_mask & ground_truth_mask)
    total = np.sum(result_mask) + np.sum(ground_truth_mask)

    dice = (2 * intersection) / total if total != 0 else 1.0
    return dice

def iou_score(result_mask, ground_truth_mask):
    result_mask = np.array(result_mask) > 0
    ground_truth_mask = np.array(ground_truth_mask) > 0

    intersection = np.sum(result_mask & ground_truth_mask)
    union = np.sum(result_mask | ground_truth_mask)

    iou = intersection / union if union != 0 else 1.0
    return iou

def precision_and_recall(result_mask, ground_truth_mask):
    result_mask = np.array(result_mask) > 0
    ground_truth_mask = np.array(ground_truth_mask) > 0

    TP = np.sum(result_mask & ground_truth_mask)
    FP = np.sum(result_mask & ~ground_truth_mask)
    FN = np.sum(~result_mask & ground_truth_mask)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 1.0

    return precision, recall



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--gt', '-g', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


# def mask_to_image(mask: np.ndarray, mask_values):
#     if isinstance(mask_values[0], list):
#         out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
#     elif mask_values == [0, 1]:
#         out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
#     else:
#         out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

#     if mask.ndim == 3:
#         mask = np.argmax(mask, axis=0)

#     for i, v in enumerate(mask_values):
#         out[mask == i] = v

#     return Image.fromarray(out)

def mask_to_image(mask: np.ndarray, mask_values, color_map=None):
    # if color map is none generate a random color map
    if color_map is None:
        # generate random color map
        color_map = {}
        for i in range(len(mask_values)):
            color_map[i] = np.random.randint(0, 255, size=3)

    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = color_map[i]
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    gt_files = args.gt
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    dice = 0
    iou = 0
    precision = 0
    recall = 0
    for i, (filename, gt_file) in enumerate(zip(in_files, gt_files)):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        gt = Image.open(gt_file).convert('L')

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            
            dice += dice_score(result, gt)
            iou += iou_score(result, gt)
            prec, rec = precision_and_recall(result, gt)
            precision += prec
            recall += rec
            
            
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
