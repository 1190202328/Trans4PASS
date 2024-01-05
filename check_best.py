import numpy as np


def check_best(path):
    mious = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            if line.find('[TEST END]') != -1:
                miou = float(line.split('mIoU:')[-1].strip())
                mious.append(miou)
    print(f'best miou={max(mious)}, best epoch={np.argmax(mious) + 1}, now epoch={len(mious) + 1}')


if __name__ == '__main__':
    root_dir = '/nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS'
    work_dir = 'workdirs/cityscapes/trans4pass_tiny_512x512'
    check_best(
        f'{root_dir}/{work_dir}/2024-01-05-11-39_Trans4PASS_trans4pass_v1_cityscape_log.txt'
    )
