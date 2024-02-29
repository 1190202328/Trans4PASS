'''
Ref: https://github.com/CharlesPikachu/mcibi
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
palette = np.split(np.array(palette), 19)

'''define the memory cfg'''
memory_cfg = {
    'num_feats_per_cls': 1,
    'feats_len': 128,
    'ignore_index': 255,
    'type': ['random_select', 'clustering'][1],
}


def init_memory(dataloader_src, dataloader_trg, backbone_net, num_classes=19, save_path=None, multi_scale=False):
    backbone_net = backbone_net.cuda()
    backbone_net.eval()
    assert memory_cfg['type'] in ['random_select', 'clustering']
    # extract feats
    FloatTensor = torch.cuda.FloatTensor
    feats_dict_s = {}
    memory_s = np.zeros((num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))

    feats_dict_t = {}
    memory_t = np.zeros((num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))

    for i in range(2):
        dataloader = [dataloader_src, dataloader_trg][i]
        feats_dict = [feats_dict_s, feats_dict_t][i]
        memory = [memory_s, memory_t][i]

        print(f'Init {i + 1}-th[total 2] memory.')
        for batch_idx, samples in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print('Processing %s/%s...' % (batch_idx + 1, len(dataloader)))
            image, gt, _, _ = samples
            image = image.type(FloatTensor)
            gt = gt.to(image.device)
            b, c, h, w = image.shape
            pred_temp = torch.zeros((b, num_classes, h, w), dtype=image.dtype).to(image.device)
            feats_temp = torch.zeros((b, memory_cfg['feats_len'], h // 4, w // 4), dtype=image.dtype).to(image.device)
            if multi_scale:
                # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # ms
                # scales = [0.75, 1.0, 1.75]  # ms
                raise Exception
            else:
                scales = [1]  # origin no scale
            for sc in scales:
                new_h, new_w = int(sc * h), int(sc * w)
                img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
                with torch.no_grad():
                    feats, pred = backbone_net(img_tem)
                    feat_fused = sum(feats)
                    pred_temp += nn.UpsamplingBilinear2d(size=(h, w))(pred)
                    feats_temp += nn.UpsamplingBilinear2d(size=(h // 4, w // 4))(feat_fused)
            pred = pred_temp / len(scales)
            feat_fused = feats_temp / len(scales)
            pred_prob = torch.softmax(pred, dim=1)
            conf, pred_cls = torch.max(pred_prob, dim=1)  # pred_cls
            pred_cls = F.interpolate(pred_cls.unsqueeze(dim=1).float(), size=feat_fused.shape[-2:], mode='nearest')
            pred_cls = pred_cls.to(image.device)
            gt = F.interpolate(gt.unsqueeze(dim=1).float(), size=feat_fused.shape[-2:], mode='nearest')
            num_channels = feat_fused.size(1)
            clsids = gt.unique()
            feat_fused = feat_fused.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
            feat_fused = feat_fused.view(-1, num_channels)  # BHW, C
            for clsid in clsids:
                clsid = int(clsid.item())
                if clsid == memory_cfg['ignore_index']: continue
                seg_cls = gt.view(-1)  # BHW
                pred_cls = pred_cls.view(-1)
                correct_feat_mask = torch.logical_and(seg_cls == clsid, pred_cls == clsid)
                if torch.count_nonzero(correct_feat_mask) < 1:
                    continue
                feats_cls = feat_fused[correct_feat_mask].mean(0).data.cpu()
                if clsid in feats_dict:
                    feats_dict[clsid].append(feats_cls.unsqueeze(0).numpy())
                else:
                    feats_dict[clsid] = [feats_cls.unsqueeze(0).numpy()]  # (19, N/Batch=num_samples, (1, feats_len))

        assert len(feats_dict) == num_classes
        for idx in range(num_classes):
            feats_cls_list = [torch.from_numpy(item) for item in feats_dict[idx]]
            feats_cls_list = torch.cat(feats_cls_list,
                                       dim=0).numpy()  # (N/Batch, feats_len) for each clsid/cluster_center
            memory[idx] = np.mean(feats_cls_list, axis=0)

    # --- joint
    memory_fused = np.zeros((num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))
    for i in range(num_classes):
        memory_fused[i] = (memory_s[i] + memory_t[i]) / 2.
    np.save(save_path, memory_fused)

    return memory_fused
