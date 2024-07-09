import torch
import yaml
from torchvision.transforms import transforms
import importlib

import utils
import main
import os
import glob
import cv2


def import_class(name):  # 动态导入指定名称的Python类
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

sparser = utils.get_parser()
p = sparser.parse_args()
# p.config = "baseline_iter.yaml"
if p.config is not None:
    with open(p.config, 'r') as f:
        try:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            default_arg = yaml.load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    sparser.set_defaults(**default_arg)
args = sparser.parse_args()  # 和baseline.yaml一样
with open(f"./configs/{args.dataset}.yaml", 'r') as f:  # 加载phoenix14.yaml
    args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

prosessor = main.Processor(args)
model, optimizer = prosessor.loading()
model.eval()


def preprocess_video(video_folder, max_frames=64, resize_shape=(224, 224)):
    # Step 1: 读取图片帧序列
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    # Step 2: 调整图片尺寸
    frames_resized = [cv2.resize(frame, resize_shape) for frame in frames]

    for i, frame in enumerate(frames):
        frame_path = os.path.join("photos/P99", f'frame_{i}.jpg')
        cv2.imwrite(frame_path, frame)

    # Step 3: 填充序列
    while len(frames_resized) < max_frames:
        # 这里简单地使用最后一帧进行填充
        frames_resized.append(frames_resized[-1])

    # Step 4: 转换为张量
    tensor_frames = torch.stack([transforms.ToTensor()(frame) for frame in frames_resized])

    # Step 5: 添加批次维度
    tensor_frames = tensor_frames.unsqueeze(0)  # 添加批次维度

    video_length = 64
    video_length = torch.LongTensor([video_length])

    return tensor_frames, video_length


tensorframes, video_length = preprocess_video("./photos/P34_s5_00_3_color")

# 输入
print(len(tensorframes.shape))
print(len(video_length))
print(tensorframes.shape)
print(video_length.shape)
print(video_length)
print("")

batch, temp, channel, height, width = tensorframes.shape

#
inputs = tensorframes.reshape(batch * temp, channel, height, width)
print(len(inputs.shape))
print(inputs.shape)

framewise1 = model.masked_bn(inputs, video_length)
print(len(framewise1.shape))
print(framewise1.shape)

framewise = framewise1.reshape(batch, temp, -1).transpose(1, 2)
print(len(framewise.shape))
print(framewise.shape)

with torch.no_grad():
    ret_dict = model(tensorframes, video_length)
total_sent = {}
total_sent = ret_dict['recognized_sents']
print(total_sent)
