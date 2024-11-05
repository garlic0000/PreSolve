import os
import shutil
import glob
import natsort
import dlib
import cv2
from tqdm import tqdm
from pathlib import Path
import yaml
from retinaface.api import Facedetecor as RetinaFaceDetector
import torch
import numpy as np


class FaceDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = RetinaFaceDetector(model_path, device)

    def cal(self, img):
        left, top, right, bottom = self.det.get_face_box(img)
        # 检测已裁剪的人脸图像 检测的参数不合法时
        left = np.clip(left, 0, img.shape[1])
        top = np.clip(top, 0, img.shape[0])
        right = np.clip(right, 0, img.shape[1])
        bottom = np.clip(bottom, 0, img.shape[0])
        return left, top, right, bottom


def get_rawpic_count(root_path):
    """
    递归地统计数据集中所有 .jpg 图片的数量
    遍历所有子目录，最终返回图片总数
    Args:
        root_path: 图片根目录

    Returns:图片总数
    """
    count = 0
    for sub in Path(root_path).iterdir():
        if sub.is_dir():
            for vid in sub.iterdir():
                if vid.is_dir():
                    # if vid.name == "30_0505funnyinnovations":
                    #     # 30_0505funnyinnovations的裁剪有问题
                    #     continue
                    count += len(glob.glob(os.path.join(
                        str(vid), "*.jpg")))
    return count


def crop_images(opt):
    """
    对数据集中的图像进行裁剪，提取出人脸区域并保存成新图片
    使用 dlib 的 HOG 面部检测器来检测每张图片中的人脸
    对人脸进行裁剪和缩放
    裁剪后的图片以 128x128 的分辨率保存
    Args:
        dataset_name:处理的数据集名称

    Returns: 已裁剪和缩放的图片

    """
    CASME_sq_rawpic_root_path = opt["CASME_sq_rawpic_root_path"]
    crop_root_path = opt["crop_root_path"]
    print(f'dataset: {opt["dataset"]}')
    face_det_model_path = opt.get("retinaface_face_detector_model_path")
    face_detector = FaceDetector(face_det_model_path)
    sum_count = get_rawpic_count(CASME_sq_rawpic_root_path)
    print("rawpic count = ", sum_count)
    if not os.path.exists(crop_root_path):
        os.makedirs(crop_root_path)

    with tqdm(total=sum_count) as tq:
        for sub_item in Path(CASME_sq_rawpic_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                # 在这里修改
                # s15 15_0101
                # casme_015,casme_015_0401
                # subject video_name
                # 将type_item改为别的
                # s15 casme_015
                # /kaggle/input/casme2/rawpic/rawpic/s15/15_0101disgustingteeth

                s_name = "casme_0{}".format(sub_item.name[1:])
                v_name = "casme_0{}".format(type_item.name[0:7])
                new_dir_path = os.path.join(
                    crop_root_path, s_name, v_name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)

                # 获取目录下所有 .jpg 文件的路径，并将它们存储在一个列表中
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                if len(img_path_list) > 0:
                    img_path_list.sort()
                    for index, img_path in enumerate(img_path_list):
                        img = cv2.imread(img_path)
                        # 对第一个图像进行剪切
                        # 将之后的图像进行对齐
                        if index == 0:
                            # 测试用
                            print(new_dir_path)
                            # h, w, c = img.shape
                            face_left, face_top, face_right, face_bottom = \
                                face_detector.cal(img)
                        img = img[face_top:face_bottom + 1, face_left:face_right + 1, :]
                        cv2.imwrite(os.path.join(
                            new_dir_path,
                            f"img_{str(index + 1).zfill(5)}.jpg"), img)
                        tq.update()


if __name__ == "__main__":
    with open("/kaggle/working/PreSolve/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    crop_images(opt)
