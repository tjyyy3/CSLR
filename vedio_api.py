from flask import Flask, request, jsonify
import cv2
import torch
import yaml
from torchvision.transforms import transforms
import importlib

import utils
import main
import os
import glob
import cv2

app = Flask(__name__)


def import_class(name):  # 动态导入指定名称的Python类
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


sparser = utils.get_parser()
p = sparser.parse_args()
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


def extract_frames(video_path, num_frames):  # 切帧
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames

    indices = sorted([int(i) for i in range(0, total_frames, total_frames // num_frames)])

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames





def preprocess_video(video_folder, max_frames=64, resize_shape=(224, 224)):
    # Step 1: 读取图片帧序列
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]
    # Step 2: 调整图片尺寸
    frames_resized = [cv2.resize(frame, resize_shape) for frame in frames]
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


def predict(vedio_tensor,lenx_tensor):
    with torch.no_grad():
        ret_dict = model(vedio_tensor, lenx_tensor)
    total_sent = {}
    total_sent = ret_dict['recognized_sents']
    print(total_sent)


@app.route('/predict', methods=['POST'])
def handle_request():
    # 遍历请求中的所有文件
    for file_storage in request.files.values():
        if file_storage and file_storage.filename.endswith('.mp4'):
            file_storage.save('uploaded_video.mp4')
            frames = extract_frames('uploaded_video.mp4', num_frames=40)
            video_tensor, len_tensor = preprocess_video(frames)
            predictions = predict(video_tensor, len_tensor)
            print(predictions)
            return jsonify({'predictions': predictions}), 200

    # 如果没有符合条件的文件，则返回错误消息
    return jsonify({'error': 'No valid video file provided'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=False)
