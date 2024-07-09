from flask import Flask, request, jsonify
import cv2
import torch
import yaml
from torchvision.transforms import transforms
import importlib
import matplotlib as plt
import utils
import main
import os
import glob
import cv2

app = Flask(__name__)

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
args = sparser.parse_args()

with open(f"./configs/{args.dataset}.yaml", 'r') as f:
    args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

processor = main.Processor(args)
model, optimizer = processor.loading()



def extract_frames(video_path, num_frames):
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
    for i, frame in enumerate(frames):
        frame_path = os.path.join("./photos/P99", f'frame_{i}.jpg')
        cv2.imwrite(frame_path, frame)

    return frames


def preprocess_video(frames, max_frames=64, resize_shape=(224, 224)):
    frames_resized = [cv2.resize(frame, resize_shape) for frame in frames]

    while len(frames_resized) < max_frames:
        frames_resized.append(frames_resized[-1])

    tensor_frames = torch.stack([transforms.ToTensor()(frame) for frame in frames_resized])
    tensor_frames = tensor_frames.unsqueeze(0)

    video_length = 64
    video_length = torch.LongTensor([video_length])

    return tensor_frames, video_length


def predict(video_tensor, len_tensor):
    model.eval()
    with torch.no_grad():
        ret_dict = model(video_tensor, len_tensor)
    total_sent = {}
    total_sent = ret_dict['recognized_sents']
    return total_sent


@app.route('/predict', methods=['POST'])
def handle_request():
    # 遍历请求中的所有文件
    for file_storage in request.files.values():
        if file_storage and file_storage.filename.endswith('.mp4'):
            file_storage.save('uploaded_video.mp4')
            frames = extract_frames('uploaded_video.mp4', num_frames=48)
            video_tensor, len_tensor = preprocess_video(frames)
            predictions = predict(video_tensor, len_tensor)
            print(predictions)
            return jsonify({'predictions': predictions}), 200

    # 如果没有符合条件的文件，则返回错误消息
    return jsonify({'error': 'No valid video file provided'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=False)  # 关闭调试模式

