import os
import glob
import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def segment_for_train(opt):
    feature_root_path = opt["feature_root_path"]
    feature_segment_root_path = opt["feature_segment_root_path"]
    anno_csv_path = opt["anno_csv_path"]
    SEGMENT_LENGTH = opt["SEGMENT_LENGTH"]  # 256帧
    RECEPTIVE_FILED = opt["RECEPTIVE_FILED"]  # 感受野 15？ 用来决定每个片段的步长
    STEP = int(RECEPTIVE_FILED // 2)

    assert os.path.exists(feature_root_path), f"{feature_root_path} not exists"
    assert os.path.exists(anno_csv_path), f"{anno_csv_path} not exists"
    if not os.path.exists(feature_segment_root_path):
        os.makedirs(feature_segment_root_path)

    anno_df = pd.read_csv(anno_csv_path)

    for sub_item in Path(feature_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        # sub_item.name 需要被替换
        # 被换成s15之类的
        # 如果原本是s15之类的就不需要换
        out_dir = os.path.join(
            feature_segment_root_path, "train", sub_item.name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        feature_path_list = glob.glob(os.path.join(str(sub_item), "*.npy"))
        # for type_item in sub_item.iterdir():
        #     if not type_item.is_dir():
        #         continue
        for feature_path in feature_path_list:
            feature_name = os.path.split(feature_path)[-1]
            video_name = os.path.splitext(feature_name)[0]
            print("feature_name")
            print(feature_name)
            print("video_name")
            print(video_name)
            # 这是
            tmp_df = anno_df[anno_df['video_name'] == video_name]
            print("tmp_tf")
            print(tmp_df)
            video_feature = np.load(feature_path)  # (t, 12, 2)
            frame_count = video_feature.shape[0]
            print("frame_count")
            print(frame_count)
            start_array = tmp_df.start_frame.values
            apex_array = tmp_df.apex_frame.values
            end_array = tmp_df.end_frame.values
            type_array = tmp_df.type_idx.values
            segment_count = frame_count // SEGMENT_LENGTH
            if frame_count - segment_count * SEGMENT_LENGTH > 0:
                segment_count += 1
            clip_boundary_list = [(i * SEGMENT_LENGTH - STEP,
                                   (i + 1) * SEGMENT_LENGTH - 1 + STEP)
                                  for i in range(segment_count)]
            feature_list = []
            labels_list = []

            # labeling every frame
            video_labels = np.zeros((frame_count, 8), dtype=int)
            for index in range(len(start_array)):
                start = start_array[index].item()
                apex = apex_array[index].item()
                end = end_array[index].item()
                ex_type = type_array[index].item()

                zero_count = 0
                if start == 0:
                    zero_count += 1
                if apex == 0:
                    zero_count += 1
                if end == 0:
                    zero_count += 1
                if zero_count >= 2:
                    continue
                # the count of optical flow might less than rgb
                if apex >= frame_count:
                    continue

                if end == 0:
                    end = (apex - start + 1) + apex - 1
                elif apex == 0:
                    apex = (end + start) // 2
                # the count of optical flow might less than rgb
                if end >= frame_count:
                    end = frame_count - 1

                # micro expression label
                if ex_type == 2:
                    video_labels[start, 0] = 1
                    video_labels[apex, 1] = 1
                    video_labels[end, 2] = 1
                    for action_index in range(start, end + 1):
                        video_labels[action_index, 3] = 1
                # macro expression label
                else:
                    video_labels[start, 4] = 1
                    video_labels[apex, 5] = 1
                    video_labels[end, 6] = 1
                    for action_index in range(start, end + 1):
                        video_labels[action_index, 7] = 1

            for clip_left_boundary, clip_right_boundary in clip_boundary_list:
                left_padding = None
                right_padding = None
                if clip_left_boundary < 0:
                    left_padding = np.zeros(
                        (abs(clip_left_boundary), 12, 2), dtype=int)
                    left_padding_label = np.zeros(
                        (abs(clip_left_boundary), 8), dtype=int)
                    clip_left_boundary = 0
                if clip_right_boundary >= frame_count:
                    right_padding = np.zeros(
                        (clip_right_boundary - frame_count + 1, 12, 2),
                        dtype=int)
                    right_padding_label = np.zeros(
                        (clip_right_boundary - frame_count + 1, 8),
                        dtype=int)
                    clip_right_boundary = frame_count - 1
                feature = video_feature[
                          clip_left_boundary: clip_right_boundary + 1]
                labels = video_labels[
                         clip_left_boundary: clip_right_boundary + 1]

                if left_padding is not None:
                    feature = np.concatenate((left_padding, feature), axis=0)
                    labels = np.concatenate(
                        (left_padding_label, labels), axis=0)
                if right_padding is not None:
                    feature = np.concatenate((feature, right_padding), axis=0)
                    labels = np.concatenate(
                        (labels, right_padding_label), axis=0)
                assert len(feature) == SEGMENT_LENGTH + STEP * 2, \
                    "time length of feature is incorrect"
                assert len(labels) == SEGMENT_LENGTH + STEP * 2, \
                    "time length of labels is incorrect"
                feature_list.append(feature)
                labels_list.append(labels)
            for index in range(len(feature_list)):
                np.savez(
                    os.path.join(
                        out_dir,
                        f"{video_name}_{str(index * SEGMENT_LENGTH).zfill(4)}"
                        f".npz"),
                    feature=feature_list[index],
                    label=labels_list[index],
                    video_name=video_name)
    print("segment for train Finished!")


def segment_for_test(opt):
    feature_root_path = opt["feature_root_path"]
    feature_segment_root_path = opt["feature_segment_root_path"]
    anno_csv_path = opt["anno_csv_path"]

    assert os.path.exists(feature_root_path), f"{feature_root_path} not exists"
    assert os.path.exists(anno_csv_path), f"{anno_csv_path} not exists"
    if not os.path.exists(feature_segment_root_path):
        os.makedirs(feature_segment_root_path)

    # 直接将训练集的数据 复制到测试集
    out_dir = os.path.join(
        feature_segment_root_path, "test")
    train_dir = os.path.join(feature_segment_root_path, "train")
    # 递归复制
    shutil.copytree(train_dir, out_dir)

    print("segment for test Finished!")


if __name__ == "__main__":
    # code_final = "/home/whcold/cold_datas/samm/samm_anno.xlsx"
    # code_final = parse_code_final(code_final)

    with open("/kaggle/working/PreSolve/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]

    segment_for_train(opt)
    segment_for_test(opt)
