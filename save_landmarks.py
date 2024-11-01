import yaml
import cv2
import dlib
from tqdm import tqdm
import glob
import csv
from pathlib import Path
import os




def get_img_count(cropped_root_path):
    """
    获取图片数量
    """
    count = 0
    for sub in Path(cropped_root_path).iterdir():
        if sub.is_dir():
            for vid in sub.iterdir():
                if vid.is_dir():
                    # 计算目录下所有 .jpg 文件的数量
                    count += len(glob.glob(os.path.join(str(vid), "*.jpg")))
    return count


def record_csv(csv_path, rows):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(rows)


def save_landmarks(opt):
    rawpic_crop_root_path = opt["rawpic_crop_root_path"]
    face_detector_model_path = opt["face_detector_model_path"]
    landmark_predictor_model_path = opt["landmark_predictor_model_path"]
    print(f'dataset: {opt["dataset"]}')
    face_detector = dlib.cnn_face_detection_model_v1(face_detector_model_path)  # 使用 GPU 加速的 CNN 人脸检测器
    face_pose_predictor = dlib.shape_predictor(landmark_predictor_model_path)
    sum_count = get_img_count(rawpic_crop_root_path)
    print("Total image count:", sum_count)

    with tqdm(total=sum_count) as tq:  # 进度条
        for sub in Path(rawpic_crop_root_path).iterdir():
            if sub.is_dir():
                for vid in sub.iterdir():
                    if vid.is_dir():
                        img_path_list = glob.glob(os.path.join(str(vid), "*.jpg"))
                        if len(img_path_list) > 0:
                            img_path_list.sort()
                            rows_landmark = []
                            csv_landmark_path = os.path.join(str(vid), "landmarks.csv")

                            for index, img_path in enumerate(img_path_list):
                                img = cv2.imread(img_path)
                                if img is None:
                                    print("该目录图片无法读取")
                                    print(img_path)
                                    continue  # 跳过无法读取的图片

                                img_height, img_width = img.shape[:2]
                                x_list, y_list = [], []
                                try:
                                    detect = face_detector(img, 1)
                                    if len(detect) == 0:
                                        print(f"该图片检测不到人脸: {img_path}")
                                        continue  # 跳过当前图片并继续

                                    shape = face_pose_predictor(img, detect[0].rect)

                                    # 提取并限制关键点的 x, y 坐标在图像范围内
                                    for i in range(68):  # 68 个面部关键点
                                        x = max(0, min(shape.part(i).x, img_width - 1))
                                        y = max(0, min(shape.part(i).y, img_height - 1))
                                        x_list.append(x)
                                        y_list.append(y)

                                    # 测试用
                                    if index == 0:
                                        dir_path = os.path.dirname(img_path)
                                        print()
                                        print("Processing directory:", dir_path)
                                except Exception as e:
                                    print(f"\nError in landmark detection for image: {img_path}")
                                    print("Error details:", e)
                                    break

                                # 保存坐标到 CSV 文件
                                rows_landmark.append(x_list + y_list)

                            record_csv(csv_landmark_path, rows_landmark)
                            tq.update(len(img_path_list))  # 更新进度条
    print('Landmark extraction completed.')


# 示例主函数调用
if __name__ == "__main__":
    with open("/kaggle/working/PreSolve/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    save_landmarks(opt)
