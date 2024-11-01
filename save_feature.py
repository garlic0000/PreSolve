import os
import glob
import csv
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import yaml
from feature_tools import calculate_roi_freature_list


def get_flow_count(root_path):
    count = 0
    for sub in Path(root_path).iterdir():
        if sub.is_dir():
            for vid in sub.iterdir():
                if vid.is_dir():
                    count += len(glob.glob(os.path.join(
                        str(vid), "flow_x*.jpg")))
    return count


def feature(opt):
    optflow_root_path = opt["optflow_root_path"]
    feature_root_path = opt["feature_root_path"]
    landmark_root_path = opt["cropped_root_path"]
    print(f'dataset: {opt["dataset"]}')
    sum_count = get_flow_count(optflow_root_path)
    print("flow count = ", sum_count)

    opt_step = 1  # int(get_micro_expression_average_len(anno_csv_path) // 2)
    print(f"opt_step: {opt_step}")

    # for debug use
    # short_video_list = []
    with tqdm(total=sum_count) as tq:
        for sub_item in Path(optflow_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                flow_x_path_list = glob.glob(
                    os.path.join(str(type_item), "flow_x*.jpg"))
                flow_y_path_list = glob.glob(
                    os.path.join(str(type_item), "flow_y*.jpg"))
                flow_x_path_list.sort()
                flow_y_path_list.sort()
                csv_landmark_path = os.path.join(
                    landmark_root_path,
                    sub_item.name, type_item.name, "landmarks.csv")
                if not os.path.exists(csv_landmark_path):
                    print("\n")
                    print(f"{csv_landmark_path} does not exist")
                    continue
                with open(csv_landmark_path, 'r') as f:
                    ior_feature_list_sequence = []  # feature in whole video
                    csv_r = list(csv.reader(f))
                    for index, row in enumerate(csv_r):
                        if index < opt_step:
                            # 用于测试
                            # print("index < opt_step")
                            # print(index, opt_step, row)
                            continue
                        i = index - opt_step
                        # # 这段有问题
                        flow_x = cv2.imread(flow_x_path_list[i],
                                            cv2.IMREAD_GRAYSCALE)
                        flow_y = cv2.imread(flow_y_path_list[i],
                                            cv2.IMREAD_GRAYSCALE)
                        flow_x_y = np.stack((flow_x, flow_y), axis=2)
                        flow_x_y = flow_x_y / np.float32(255)
                        flow_x_y = flow_x_y - 0.5
                        landmarks = np.array(
                            [(int(row[index]), int(row[index + 68]))
                             for index in range(int(len(row) // 2))])

                        try:
                            # radius=5 从面部关键点半径为5的区域提取感兴趣区域ROI
                            ior_feature_list = calculate_roi_freature_list(
                                flow_x_y, landmarks, radius=5)
                            ior_feature_list_sequence.append(
                                np.stack(ior_feature_list, axis=0))
                            tq.update()
                        except Exception as exp:
                            ior_feature_list_sequence = []
                            print("ior_feature_list 有问题")
                            print(f"{sub_item.name}  {type_item.name}")
                            # 打印异常信息
                            print(str(exp))
                            break
                    if len(ior_feature_list_sequence) > 0:
                        new_type_dir_path = os.path.join(
                            feature_root_path, sub_item.name)
                        if not os.path.exists(new_type_dir_path):
                            os.makedirs(new_type_dir_path)
                        np.save(os.path.join(
                            new_type_dir_path, f"{type_item.name}.npy"),
                            np.stack(ior_feature_list_sequence, axis=0))


if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]

    feature(opt)
