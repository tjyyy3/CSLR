import os
import cv2


def save_images_from_video(video_path, framecut_path):  # folder_path -> video_path
    capture = cv2.VideoCapture(video_path)

    # fps = capture.get(cv2.CAP_PROP_FPS)
    fps_all = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps_all is %d" % fps_all)

    folder_path = os.path.join(framecut_path, video_path[-21:-4])

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    k = 0
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        if k > 0:
            image_path = os.path.join(folder_path, '{:06d}.jpg'.format(k))
            cv2.imwrite(image_path, frame)

        k = k + 1

    capture.release()
    print("帧获取完成 -- {0}".format(video_path))


def load_videos(data_path, framecut_path):
    sent_folders = sorted(os.listdir(data_path))
    sent_paths = [os.path.join(data_path, p) for p in sent_folders]

    for sent_path in sent_paths:
        selected_folders = [os.path.join(sent_path, v) for v in sorted(os.listdir(sent_path)) if
                            v[-4:] in ['.avi']]  # 加入一个判断，只读取avi视频文件

        [save_images_from_video(f, framecut_path) for f in selected_folders]


if __name__ == '__main__':
    data_path = "E:\\BaiduNetdiskDownload\\SLR_Dataset\\Continuous_SLR_dataset\\color"
    framecut = "E:\\pythonfiles\\SLR_graduation_Project\\framecut"
    load_videos(data_path, framecut)
