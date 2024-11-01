import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import os
import glob
import yaml




def pol2cart(rho, phi):
    """
    Args:
        rho:
        phi:

    Returns:输出是 u 和 v 分量，而不是原始极坐标

    """
    u = rho * np.cos(phi)
    v = rho * np.sin(phi)
    return u, v


def calculate_tvl1_optical_flow(frame1, frame2):
    """

    Args:
        frame1: 前一帧
        frame2: 当前帧

    Returns: 原始光流

    """
    # Compute Optical Flow Features
    # 使用gpu
    optical_flow = cv2.cuda.OpticalFlowDual_TVL1.create()
    gpu_img1 = cv2.cuda.GpuMat()
    gpu_img2 = cv2.cuda.GpuMat()
    gpu_img1.upload(frame1)
    gpu_img2.upload(frame2)
    # flow = optical_flow.calc(img1, img2, None)
    flow = optical_flow.calc(gpu_img1, gpu_img2, None)
    flow = flow.download()  # 从 GPU 下载到 CPU
    return flow


def save_flow_to_image(flow, paths, sub_name, vid_name, frame_index):
    """
    保存光流图片 供下一步分析
    Args:
        flow: 提取的原始光流
        savepaths: 保存路径集合
        frame_index: 帧编号

    Returns:

    """
    # 保存路径
    optflow_xy_root_path, optflow_ma_root_path, optflow_uv_root_path = paths
    savepath_xy = os.path.join(optflow_xy_root_path, sub_name, vid_name)
    savepath_ma = os.path.join(optflow_ma_root_path, sub_name, vid_name)
    savepath_uv = os.path.join(optflow_uv_root_path, sub_name, vid_name)

    if not os.path.exists(savepath_xy):
        os.makedirs(savepath_xy)
    if not os.path.exists(savepath_ma):
        os.makedirs(savepath_ma)
    if not os.path.exists(savepath_uv):
        os.makedirs(savepath_uv)

    # 转换为极坐标
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 转换成直角坐标
    u, v = pol2cart(magnitude, angle)

    # 将 flow[..., 0] 和 flow[..., 1] 归一化并转换为灰度图
    x_flow_norm = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    y_flow_norm = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    gray_x_flow = x_flow_norm.astype(np.uint8)
    gray_y_flow = y_flow_norm.astype(np.uint8)
    # 保存为图像文件
    cv2.imwrite(os.path.join(savepath_xy, f"flow_x_{frame_index:05d}.jpg"), gray_x_flow)
    cv2.imwrite(os.path.join(savepath_xy, f"flow_y_{frame_index:05d}.jpg"), gray_y_flow)

    # 生成 magnitude angle 灰度图
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    angle_norm = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)
    gray_magnitude = magnitude_norm.astype(np.uint8)
    gray_angle = angle_norm.astype(np.uint8)
    # 保存为图像文件
    cv2.imwrite(os.path.join(savepath_ma, f"flow_x_{frame_index:05d}.jpg"), gray_magnitude)
    cv2.imwrite(os.path.join(savepath_ma, f"flow_y_{frame_index:05d}.jpg"), gray_angle)

    # 将 u 和 v 归一化并转换为灰度图
    u_norm = cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX)
    v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    gray_u = u_norm.astype(np.uint8)
    gray_v = v_norm.astype(np.uint8)
    # 保存为图像文件
    cv2.imwrite(os.path.join(savepath_uv, f"flow_x_{frame_index:05d}.jpg"), gray_u)
    cv2.imwrite(os.path.join(savepath_uv, f"flow_y_{frame_index:05d}.jpg"), gray_v)


def process_optical_flow_for_dir(input_dir, paths, sub_name, vid_name):
    """
    为一个路径下的图片帧计算光流
    每相邻两帧计算一次光流
    Args:
        input_dir: 计算光流的图片帧路径 或者一段视频的路径
        savepaths: 保存路径合集

    Returns:

    """
    image_list = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if len(image_list) < 2:
        print(f"Not enough images in {input_dir} to compute optical flow")
        return

    frame_index = 0
    prev_frame = cv2.imread(image_list[0], cv2.IMREAD_GRAYSCALE)

    for i in range(1, len(image_list)):
        frame = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
        flow = calculate_tvl1_optical_flow(prev_frame, frame)
        save_flow_to_image(flow, paths, sub_name, vid_name, frame_index)
        frame_index += 1
        prev_frame = frame


def get_rawpic_crop_count(crop_root_path):
    """
        遍历所有子目录，最终返回子目录总数
        Args:
            root_path: 已裁剪图片根目录

        Returns:图片总数
    """
    count = 0
    for sub in Path(crop_root_path).iterdir():
        if sub.is_dir():
            for vid in sub.iterdir():
                if vid.is_dir():
                    if len(glob.glob(
                            os.path.join(str(vid), "*.jpg"))) > 0:
                        count += 1
    return count


def optflow(opt):
    """
    Returns:

    """
    rawpic_croped_root_path = opt["rawpic_croped_root_path"]
    optflow_xy_root_path = opt["optflow_xy_root_path"]
    optflow_ma_root_path = opt["optflow_ma_root_path"]
    optflow_uv_root_path = opt["optflow_uv_root_path"]
    print(f'dataset: {opt["dataset"]}')
    if not os.path.exists(optflow_xy_root_path):
        os.makedirs(optflow_xy_root_path)
    if not os.path.exists(optflow_ma_root_path):
        os.makedirs(optflow_ma_root_path)
    if not os.path.exists(optflow_uv_root_path):
        os.makedirs(optflow_uv_root_path)
    paths = optflow_xy_root_path, optflow_ma_root_path, optflow_uv_root_path

    dir_count = get_rawpic_crop_count(rawpic_croped_root_path)
    print("flow count = ", dir_count)

    with tqdm(total=dir_count) as tq:
        for sub in Path(rawpic_croped_root_path).iterdir():
            if sub.is_dir():
                for vid in sub.iterdir():
                    if vid.is_dir():
                        print()
                        print(f"Processing optical flow for {vid}")
                        process_optical_flow_for_dir(str(vid), paths, sub.name, vid.name)
                        tq.update()


if __name__ == "__main__":
    with open("/kaggle/working/PreSolve/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    optflow(opt)
