from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import os
import uuid
import torch
import numpy as np

import slr_network

app = Flask(__name__)
api = Api(app)

# 初始化模型并将其加载到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = slr_network.SLRModel()
model.load_state_dict(torch.load('your_model_weights.pth'))
model.eval()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 处理视频的帧数
frame_sequence = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame):
    # 这里是你的自定义预处理逻辑，可以根据需要进行修改
    # 例如，调整大小（resize）、正规化等操作
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return frame

class VideoUpload(Resource):
    def post(self):
        # 检查文件是否存在于请求中
        if 'file' not in request.files:
            return {'message': 'No file part'}, 400

        file = request.files['file']

        # 检查文件名是否合法
        if file.filename == '':
            return {'message': 'No selected file'}, 400

        if file and allowed_file(file.filename):
            # 为上传的视频生成唯一的文件名
            file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'mp4'
            filename = str(uuid.uuid4()) + '.' + file_extension

            # 保存视频文件到指定目录
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # 切帧并保存到指定目录
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            frame_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'frames', filename.split('.')[0])

            os.makedirs(frame_save_path, exist_ok=True)

            # 使用OpenCV读取视频文件
            cap = cv2.VideoCapture(video_path)

            # 获取视频帧率
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 指定每秒切多少帧
            frames_per_second = 5 # 可以根据需要调整

            # 计算每隔多少帧采样一次
            frame_interval = int(fps / frames_per_second)

            # 循环读取视频帧并保存为图像文件
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 仅保存每隔 frame_interval 帧的帧
                if count % frame_interval == 0:
                    frame_filename = f"{count}.jpg"
                    frame_filepath = os.path.join(frame_save_path, frame_filename)
                    cv2.imwrite(frame_filepath, frame)

                count += 1

            cap.release()

            return {'message': 'Video uploaded and frames saved successfully'}, 201

        else:
            return {'message': 'Invalid file type'}, 400

# 添加资源到API
api.add_resource(VideoUpload, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=True)
