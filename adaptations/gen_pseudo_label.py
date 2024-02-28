import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils import data

from compute_iou import fast_hist, per_class_iu
from dataset.densepass_dataset import densepassDataSet, densepassTestDataSet
from model.trans4passplus import Trans4PASS_plus_v1, Trans4PASS_plus_v2
from utils.init import set_random_seed, freeze_model

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
EMB_CHANS = 128
DATA_DIRECTORY = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/DensePASS/DensePASS'
DATA_LIST_PATH = './dataset/densepass_list/train.txt'
DATA_LIST_TEST_PATH = './dataset/densepass_list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 1
NUM_WORKERS = 0
MODEL = 'Trans4PASS_plus_v2'
SET = 'train'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice Deeplab.")
    parser.add_argument("--emb-chans", type=int, default=EMB_CHANS,
                        help="Number of channels in decoder head.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--multi-scale", action='store_true')
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    # set random seed
    set_random_seed(args.random_seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'Trans4PASS_plus_v1':
        model = Trans4PASS_plus_v1(num_classes=args.num_classes, emb_chans=args.emb_chans)
    elif args.model == 'Trans4PASS_plus_v2':
        model = Trans4PASS_plus_v2(num_classes=args.num_classes, emb_chans=args.emb_chans)
    else:
        raise ValueError
    print(f'loading from [{args.restore_from}]')
    print(f'saving into [{args.save}]')
    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    print(msg)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)
    freeze_model(model)
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    targetset = densepassDataSet(args.data_dir, args.data_list, crop_size=(2048, 400), set=args.set)
    testloader = data.DataLoader(targetset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=False)
    interp = nn.Upsample(size=(400, 2048), mode='bilinear', align_corners=True)

    # eval on dp test set
    real_test_set = densepassTestDataSet(args.data_dir, DATA_LIST_TEST_PATH, crop_size=(2048, 400),
                                         mean=IMG_MEAN, scale=False, mirror=False, set='val')
    real_testloader = data.DataLoader(real_test_set, batch_size=1, shuffle=False, pin_memory=True)
    print('eval on dp test set...')
    hist = np.zeros((args.num_classes, args.num_classes))
    for index, batch in enumerate(real_testloader):
        image, label, _, name = batch
        with torch.no_grad():
            _, output2 = model(image.to(device))
        output = output2.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), args.num_classes)
    mIoUs = per_class_iu(hist)
    for ind_class in range(args.num_classes):
        temp_str = '===>{:<15}:\t{}'.format(ind_class, str(round(mIoUs[ind_class] * 100, 2)))
        print(temp_str)
    mIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(mIoU))

    # generate pseudo labels for dp train set
    print('generate pseudo labels for dp train set...')
    predicted_label = np.zeros((len(targetset), 400, 2048), dtype=np.int8)
    predicted_prob = np.zeros((len(targetset), 400, 2048), dtype=np.float16)
    image_name = []

    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print('{}/{} processed'.format(index, len(testloader)))

        image, _, name = batch
        image_name.append(name[0])
        image = image.to(device)
        b, c, h, w = image.shape
        output_temp = torch.zeros((b, NUM_CLASSES, h, w), dtype=image.dtype).to(device)
        if args.multi_scale:
            # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # ms
            # scales = [0.75, 1.0, 1.75]  # ms
            raise Exception
        else:
            scales = [1]  # origin no scale
        for sc in scales:
            new_h, new_w = int(sc * h), int(sc * w)
            img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
            with torch.no_grad():
                _, output = model(img_tem)
                output_temp += interp(output)
        output = output_temp / len(scales)
        output = F.softmax(output, dim=1)
        output = interp(output).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label
        predicted_prob[index] = prob
    thres = []
    for i in range(NUM_CLASSES):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x) * 0.5))])
    print(thres)
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    print(thres)
    for index in range(len(targetset) // BATCH_SIZE):
        name = image_name[index]
        label = predicted_label[index]
        label = np.asarray(label, dtype=np.uint8)
        prob = predicted_prob[index]
        for i in range(NUM_CLASSES):
            label[(prob < thres[i]) * (label == i)] = 255
        output = label
        output = Image.fromarray(output)
        name = name.replace('.jpg', '_labelTrainIds.png')
        save_fn = os.path.join(args.save, name)
        if not os.path.exists(os.path.dirname(save_fn)):
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        output.save(save_fn)


if __name__ == '__main__':
    main()
