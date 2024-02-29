import argparse
import glob
import logging
import os
import os.path as osp
import sys
import time

os.chdir(sys.path[0])
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils import data

from compute_iou import fast_hist, per_class_iu
from dataset.cs_dataset_src import CSSrcDataSet
from dataset.densepass_dataset import densepassDataSet, densepassTestDataSet
from model.discriminator import FCDiscriminator
from model.trans4passplus import Trans4PASS_plus_v1, Trans4PASS_plus_v2
from utils.init import set_random_seed, freeze_model, unfreeze_model
from model.memory import init_memory
from utils.loss import feat_kl_loss

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'Trans4PASS_plus_v2'
EMB_CHANS = 128
BATCH_SIZE = 2
NUM_WORKERS = BATCH_SIZE * 2
SOURCE_NAME = 'CS'
DATA_DIRECTORY = '/nfs/s3_common_dataset/cityscapes'
DATA_LIST_PATH = 'dataset/cityscapes_list/train.txt'
TARGET_NAME = 'DP'

IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/DensePASS/DensePASS'
DATA_LIST_PATH_TARGET = 'dataset/densepass_list/train.txt'
DATA_LIST_PATH_TARGET_TEST = 'dataset/densepass_list/val.txt'
INPUT_SIZE_TARGET = '2048,400'
TARGET_TRANSFORM = 'resize'
INPUT_SIZE_TARGET_TEST = '2048,400'
LEARNING_RATE = 2.5e-6
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 100000
NUM_STEPS_STOP = int(NUM_STEPS * 0.8)  # early stopping
NUM_PROTOTYPE = 50
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 250
DIR_NAME = 'my_{}2{}_{}_MPA_'.format(SOURCE_NAME, TARGET_NAME, MODEL)
SNAPSHOT_DIR = '/nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/snapshots/' + DIR_NAME
WEIGHT_DECAY = 0.0005
# LOG_DIR = './log'
LOG_DIR = SNAPSHOT_DIR

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_TARGET = 0.001

# ---- memory
MOMENTUM_MEM = 0.999
ITER_UPDATE_MEM = 100
# --- pseudo label
LAMBDA_SSL = 1
LAMBDA_KL_S = 0.001
LAMBDA_KL_T = 0.001
TARGET = 'densepass'
SET = 'train'

NAME_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : Trans4PASS_v1, Trans4PASS_v2")
    parser.add_argument("--emb-chans", type=int, default=EMB_CHANS,
                        help="Number of channels in decoder head.")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-target-test", type=str, default=DATA_LIST_PATH_TARGET_TEST,
                        help="Path to the file listing the images in the target val dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--num-prototype", type=int, default=NUM_PROTOTYPE,
                        help="Number of prototypes.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default='',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default='',
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--continue-train", action="store_true",
                        help="continue training")
    parser.add_argument("--ssl-dir", type=str)
    return parser.parse_args()


args = get_arguments()


def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logging.root.name = name
    logging.root.setLevel(logging.INFO)
    # don't log results for the non-master process
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.root.addHandler(fh)
    # else:
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logging.root.addHandler(ch)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""
    # set random seed
    set_random_seed(args.random_seed)

    # change args
    exp_name = args.snapshot_dir
    args.snapshot_dir = SNAPSHOT_DIR + exp_name
    args.log_dir = LOG_DIR + exp_name
    init_memory_path = f'{args.snapshot_dir}/init_memory.npy'
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    setup_logger('Trans4PASS', args.log_dir, f'{TIME_STAMP}_log.txt')

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    w, h = map(int, INPUT_SIZE_TARGET_TEST.split(','))
    input_size_target_test = (w, h)

    Iter = 0
    bestIoU = 0
    mIoU = 0

    # Create network
    # init G
    if args.model == 'Trans4PASS_plus_v1':
        model = Trans4PASS_plus_v1(num_classes=args.num_classes, emb_chans=args.emb_chans)
    elif args.model == 'Trans4PASS_plus_v2':
        model = Trans4PASS_plus_v2(num_classes=args.num_classes, emb_chans=args.emb_chans)
    else:
        raise ValueError
    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']

    msg = model.load_state_dict(saved_state_dict, strict=False)
    logging.info(msg)

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes).to(device)

    if not os.path.exists(init_memory_path):
        source_trainset_temp = CSSrcDataSet(args.data_dir, args.data_list, crop_size=input_size, set='train')
        source_trainloader_temp = data.DataLoader(source_trainset_temp, batch_size=1, shuffle=False)
        target_trainset_temp = densepassDataSet(args.data_dir_target, args.data_list_target,
                                                crop_size=input_size_target,
                                                set='train',
                                                ssl_dir=args.ssl_dir)
        target_trainloader_temp = data.DataLoader(target_trainset_temp, batch_size=1, shuffle=False)
        init_mem = init_memory(source_trainloader_temp, target_trainloader_temp, model, num_classes=args.num_classes,
                               save_path=init_memory_path)
        del source_trainset_temp, source_trainloader_temp, target_trainset_temp, target_trainloader_temp
    else:
        init_mem = np.load(init_memory_path)
    init_mem = torch.from_numpy(init_mem).to(device).to(torch.get_default_dtype())
    init_batch_mem = [[] for _ in range(NUM_CLASSES)]

    unfreeze_model(model)
    model.to(device)

    unfreeze_model(model_D)
    model_D.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    else:
        pass

    # init data loader
    trainset = CSSrcDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                            crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                            set=args.set)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    # --- SSL_DIR
    targetset = densepassDataSet(args.data_dir_target, args.data_list_target,
                                 max_iters=args.num_steps * args.batch_size,
                                 crop_size=input_size_target, scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                 set=args.set,
                                 ssl_dir=args.ssl_dir, trans=TARGET_TRANSFORM)
    targetloader = data.DataLoader(targetset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    logging.info('\n--- load TEST dataset ---')

    # test_h, test_w = 400, 2048
    test_w, test_h = input_size_target_test
    targettestset = densepassTestDataSet(args.data_dir_target, args.data_list_target_test, crop_size=(test_w, test_h),
                                         mean=IMG_MEAN, scale=False, mirror=False, set='val')
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)

    # init optimizer
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    # init loss
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.8149201869965
    weight[1] = 6.9850029945374
    weight[2] = 3.7890393733978
    weight[3] = 9.9428062438965
    weight[4] = 9.7702074050903
    weight[5] = 9.5110931396484
    weight[6] = 10.311357498169
    weight[7] = 10.026463508606
    weight[8] = 4.6323022842407
    weight[9] = 9.5608062744141
    weight[10] = 7.8698215484619
    weight[11] = 9.5168733596802
    weight[12] = 10.373730659485
    weight[13] = 6.6616044044495
    weight[14] = 10.260489463806
    weight[15] = 10.287888526917
    weight[16] = 10.289801597595
    weight[17] = 10.405355453491
    weight[18] = 10.138095855713
    # weight[19] = 0
    weight = weight.to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
    seg_loss_target = torch.nn.CrossEntropyLoss(ignore_index=255)
    L1_loss = torch.nn.L1Loss(reduction='none')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

    # start training
    for i_iter in range(Iter, args.num_steps):

        loss_seg_value = 0
        loss_seg_value_t = 0
        loss_adv_target_value = 0
        loss_D_value = 0
        loss_kl_s_value = 0
        loss_kl_t_value = 0

        # reset optimizer
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # get data
        _, batch_source = trainloader_iter.__next__()
        images_source, labels_source, _, _ = batch_source
        images_source = images_source.to(device)
        labels_source = labels_source.long().to(device)

        _, batch_target = targetloader_iter.__next__()
        images_target, labels_target, _, _ = batch_target
        images_target = images_target.to(device)
        labels_target = labels_target.long().to(device)

        # train G
        freeze_model(model_D)

        # train with source
        src_features, pred_source = model(images_source)  # src_feature = [c1, c2, c3, c4]

        src_feature = sum(src_features)
        loss_kl_feat_src, batch_feats_mem_s, select_feat_src = feat_kl_loss(src_feature, labels_source, init_mem)

        loss_seg = seg_loss(pred_source, labels_source)
        loss = loss_seg + loss_kl_feat_src * LAMBDA_KL_S

        # proper normalization
        loss.backward()
        loss_seg_value += loss_seg.item()
        loss_kl_s_value += loss_kl_feat_src.item()

        # === train with target
        trg_features, pred_target = model(images_target)

        trg_feature = sum(trg_features)
        loss_kl_feat_trg, batch_feats_mem_t, select_feat_trg = feat_kl_loss(trg_feature, labels_target, init_mem)
        loss_seg_trg = seg_loss_target(pred_target, labels_target)
        D_out = model_D(F.softmax(pred_target, dim=1))
        loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss = loss_seg_trg * LAMBDA_SSL + loss_kl_feat_trg * LAMBDA_KL_T + args.lambda_adv_target * loss_adv_target
        loss.backward()
        loss_seg_value_t += loss_seg_trg.item()
        loss_adv_target_value += loss_adv_target.item()
        loss_kl_t_value += loss_kl_feat_trg.item()

        # === train D
        unfreeze_model(model_D)

        # train with source
        pred_source = pred_source.detach()
        D_out = model_D(F.softmax(pred_source, dim=1))

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value += loss_D.item()

        # train with target
        pred_target = pred_target.detach()
        D_out = model_D(F.softmax(pred_target, dim=1))

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value += loss_D.item()

        for clsid in range(NUM_CLASSES):
            feat_cls_s = batch_feats_mem_s[clsid].cpu().data.numpy()
            feat_cls_t = batch_feats_mem_t[clsid].cpu().data.numpy()
            if feat_cls_s.mean() != 0:
                init_batch_mem[clsid].append(batch_feats_mem_s[clsid].cpu().data.numpy())
            if feat_cls_t.mean() != 0:
                init_batch_mem[clsid].append(batch_feats_mem_t[clsid].cpu().data.numpy())
        if i_iter % ITER_UPDATE_MEM == 0 and i_iter > 0:
            cluster_batch_feats_mem_s = torch.zeros_like(init_mem)
            for clsid in range(NUM_CLASSES):
                if len(init_batch_mem[clsid]) > 1:
                    batch_center = np.mean(init_batch_mem[clsid])[None, ...]
                    cluster_batch_feats_mem_s[clsid] = torch.from_numpy(batch_center).to(
                        cluster_batch_feats_mem_s.dtype)
            # ema
            init_mem = init_mem * MOMENTUM_MEM + cluster_batch_feats_mem_s * (1 - MOMENTUM_MEM)

        optimizer.step()
        optimizer_D.step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg_value,
                'loss_seg_t': loss_seg_value_t,
                "loss_kl_s_value": loss_kl_s_value,
                "loss_kl_t_value": loss_kl_t_value,
                'loss_adv_D': loss_adv_target_value,
                'loss_D': loss_D_value,
                'miou_T': mIoU
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)
        if i_iter % 10 == 0:
            logging.info(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_seg_t = {7:.3f}, loss_adv = {3:.3f} loss_D = {4:.3f}, l_kl_s = {5:.3f}, l_kl_t = {6:.3f}'.format(
                    i_iter, args.num_steps, loss_seg_value, loss_adv_target_value, loss_D_value, loss_kl_s_value,
                    loss_kl_t_value, loss_seg_value_t))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            logging.info('taking snapshot ...')
            freeze_model(model)
            hist = np.zeros((args.num_classes, args.num_classes))
            for index, batch in enumerate(testloader):
                image, label, _, name = batch
                with torch.no_grad():
                    _, output2 = model(Variable(image).to(device))
                output = output2.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                label = label.cpu().data[0].numpy()
                hist += fast_hist(label.flatten(), output.flatten(), args.num_classes)
            best_miou_str = '\n' + '-' * 10 + '\n'
            mIoUs = per_class_iu(hist)
            for ind_class in range(args.num_classes):
                temp_str = '===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2)))
                logging.info(temp_str)
                best_miou_str += f'{temp_str}\n'
            mIoU = round(np.nanmean(mIoUs) * 100, 2)
            logging.info('===> mIoU: ' + str(mIoU))
            best_miou_str += f'best miou = {mIoU}, best iter = {i_iter}\n'
            if mIoU >= bestIoU:
                bestIoU = mIoU
                pre_filename = osp.join(args.snapshot_dir + 'best*.pth')
                pre_filename = glob.glob(pre_filename)
                try:
                    for p in pre_filename:
                        os.remove(p)
                except OSError as e:
                    logging.info(e)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best.pth'))
                torch.save(model_D.state_dict(),
                           osp.join(args.snapshot_dir, 'best_D.sh'))
                with open(osp.join(args.snapshot_dir, 'best_miou.txt'), mode='w', encoding='utf-8') as f:
                    f.write(best_miou_str)
            unfreeze_model(model)

        if i_iter >= args.num_steps_stop:
            logging.info('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, 'latest.pth'))
            torch.save(model_D.state_dict(),
                       osp.join(args.snapshot_dir, 'latest_D.pth'))
            break

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
