import re
import os
import cv2
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import random


def str_split(lable):   #在字符串中间加上空格
    s1 = []
    for i in lable:
        s1.append(i)
    str = " ".join(s1)
    return str

def csv2dict(anno_path):
    info_dict = dict()
    temp_info_dict = dict()
    test_dict = dict()
    train_dict = dict()

    temp_info_dict['prefix'] = "/home/czk/Data/SLR_Dataset/CSL_Continuous/color"
    num = 0
    s = 0
    inputs_list = pandas.read_table(anno_path, header=None)
    for index, file_info in inputs_list.iterrows():       #iterrows 按行遍历
        if index % 10 == 0:
            s = s+1
        file_info = file_info.to_list()[0]
        fileid, label = file_info.split(" ")
        label = str_split(label)
        for a in range(1, 51):
            for b in range(5):
                folder = f"P{'%02d' % a}_s{s}_0{fileid[-1]}_{b}_color/*.jpg"
                num_frames = len(glob.glob(f"{temp_info_dict['prefix']}/{fileid}/{folder}"))
                info_dict[num] = {
                    'fileid': fileid,
                    'folder': folder,
                    'label': label,
                    'num_frames': num_frames,
                }
                num = num+1
    tra_dict = info_dict.copy()
    length = len(info_dict)
    random.seed(0)      #固定随机种子
    tes = random.sample(info_dict.keys(), int(0.2*length))
    x = 0
    for key in tes:
        test_dict[x] = info_dict.get(key)
        test_dict[x]['folder'] = f"test/{test_dict[x]['fileid']}/{test_dict[x]['folder']}"
        x = x+1
        del tra_dict[key]
    xx = 0
    for item in tra_dict.values():
        item['folder'] = f"train/{item['fileid']}/{item['folder']}"
        train_dict[xx] = item
        xx = xx+1
    del tra_dict

    info_dict['prefix'] = "/home/czk/Data/SLR_Dataset/CSL_Continuous/color"
    train_dict['prefix'] = "/home/czk/Data/SLR_Dataset/CSL_Continuous/color"
    test_dict['prefix'] = "/home/czk/Data/SLR_Dataset/CSL_Continuous/color"
    return info_dict, train_dict, test_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']}|{v['folder']}|{v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split( )
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


# def resize_img(img_path, dsize='210x260px'):
#     dsize = tuple(int(res) for res in re.findall("\d+", dsize))
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
#     return img

def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    sp = img.shape  #获取图像形状：返回【行数值，列数值】列表
    sz1 = sp[0]     #图像的高度（行 范围）
    sz2 = sp[1]     #图像的宽度（列 范围）
    # 你想对文件的操作
    a = int(240)
    b = int(720)
    c = int(sz2 / 2 - 240)
    d = int(sz2 / 2 + 240)
    cropImg = img[a:b, c:d]  # 裁剪图像
    img = cv2.resize(cropImg, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img

def resize_dataset(video_idx, dsize, info_dict, data_type):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['fileid']}/{info['folder'].split('/', 2)[2]}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        if data_type == 'train':
            rs_img_path = img_path.replace("color", "train", 1)
        elif data_type == 'test':
            rs_img_path = img_path.replace("color", "test", 1)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='SLR_dataset',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='../dataset/phoenix2014/phoenix-2014-multisigner',
                        help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default='annotations/manual/{}.corpus.csv',
                        help='annotation prefix')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()

    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")

    # generate information dict

    information, train, test = csv2dict(f"/home/czk/Data/SLR_Dataset/CSL_Continuous/corpus.txt")
    np.save(f"./SLR_dataset/train_info.npy", train)
    np.save(f"./SLR_dataset/test_info.npy", test)
    # update the total gloss dict
    sign_dict_update(sign_dict, information)
    # generate groudtruth stm for evaluation
    generate_gt_stm(train, f"./SLR_dataset/{args.dataset}-groundtruth-train.stm")
    generate_gt_stm(test, f"./SLR_dataset/{args.dataset}-groundtruth-test.stm")
    # resize images
    video_train_index = [i for i in train.keys()]
    del video_train_index[-1]   #因为最后一个元素是['prefix']的内容
    print(f"Resize image to {args.output_res}")
    if True:
        if args.multiprocessing:
            run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=train, data_type="train"), video_train_index)
        else:
            t = tqdm(video_train_index)
            for idx in t:
                run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=train, data_type="train"), idx)
    video_test_index = [i for i in test.keys()]
    del video_test_index[-1]   #因为最后一个元素是['prefix']的内容
    if True:
        if args.multiprocessing:
            run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=test, data_type="test"), video_test_index)
        else:
            tt = tqdm(video_test_index)
            for idxx in tqdm(video_test_index):
                run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=test, data_type="test"), idxx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
    print("成功")
