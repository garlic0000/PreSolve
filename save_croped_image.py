import os
import shutil
import glob
import natsort
import dlib
import cv2
from tqdm import tqdm
from pathlib import Path
import yaml

print(dlib.DLIB_USE_CUDA)  # 输出 True 则表示启用了 GPU 支持



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
    rawpic_crop_root_path = opt["rawpic_crop_root_path"]
    face_detector_model_path = opt["face_detector_model_path"]
    print(f'dataset: {opt["dataset"]}')
    # # 加载 HOG 人脸检测器
    # face_detector = dlib.get_frontal_face_detector()
    # 使用cnn_face_detector代替face_detector
    cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_model_path)
    sum_count = get_rawpic_count(CASME_sq_rawpic_root_path)
    print("rawpic count = ", sum_count)
    with tqdm(total=sum_count) as tq:  # 进度条
        # s15, s16, s19等subject目录名称
        for sub in Path(CASME_sq_rawpic_root_path).iterdir():
            if not sub.is_dir():
                continue
            # 把裁剪的图片保存至 'rawpic_crop'
            # 创建新目录 'rawpic_crop'
            if not os.path.exists(rawpic_crop_root_path):
                os.mkdir(rawpic_crop_root_path)

            # 为每一个subject创建目录
            # # /kaggle/input/casme2/rawpic/rawpic/s15/15_0101disgustingteeth
            #
            #                 s_name = "casme_0{}".format(sub_item.name[1:])
            #                 v_name = "casme_0{}".format(type_item.name[0:7])
            s_name = "casme_0{}".format(sub.name[1:])
            dir_crop_sub = os.path.join(rawpic_crop_root_path, s_name)
            if not os.path.exists(dir_crop_sub):
                os.mkdir(dir_crop_sub)
            print()  # 输出一个空行
            print('Subject', sub.name)
            for vid in sub.iterdir():
                if not vid.is_dir():
                    continue
                print()  # 输出一个空行
                print("Video", vid.name)
                # 为每段视频创建目录
                v_name = "casme_0{}".format(vid.name[0:7])
                dir_crop_sub_vid = os.path.join(dir_crop_sub, v_name)
                if not os.path.exists(dir_crop_sub_vid):
                    os.mkdir(dir_crop_sub_vid)
                # natsort 是一个第三方库，用于执行“自然排序”，
                # 也就是按人类习惯的方式进行排序。
                # 例如，按自然顺序，img2.jpg 会排在 img10.jpg 前面，而不是后面。
                dir_crop_sub_vid_img_list = glob.glob(os.path.join(str(vid), "*.jpg"))
                # 读取每张图片
                for dir_crop_sub_vid_img in natsort.natsorted(dir_crop_sub_vid_img_list):
                    img = os.path.basename(dir_crop_sub_vid_img)  # 获取文件名，例如 'img001.jpg'
                    img_name = img[3:-4]  # 获取 '001'
                    # 读入图片
                    image = cv2.imread(dir_crop_sub_vid_img)
                    detected_faces = cnn_face_detector(image, 1)
                    if img_name == '001':
                        # 使用第一帧（图片名为 001）的面部作为参考框架来确定面部的裁剪边界
                        # 后续帧中将使用同样的边界
                        for face_rect in detected_faces:
                            # cnn_face_detector(image, 1) 返回的是包含 dlib.mmod_rectangle 对象的列表
                            # 使用 face_rect.rect.top()等来访问面部边界框的位置
                            face_top = face_rect.rect.top()
                            face_bottom = face_rect.rect.bottom()
                            face_left = face_rect.rect.left()
                            face_right = face_rect.rect.right()
                    # casme_030_0505 casme_026_0101 在进行人脸裁剪之后分别有111张和2张图片检测不到
                    # 可能需要进行填充
                    # 关于casme_030_0505这部分 检测到人脸的部分是正常的 没检测到的部分是头部发生了偏移
                    # 由于统一裁剪导致的 未检测到人脸的图片中 左边几乎有一半的人脸在外面 可能需要加一些
                    if v_name == "casme_030_0505":
                        face_left = face_left - 5
                    # 关于casme_026_0101这部分 右边多裁剪了一部分 在右边增加一些
                    if v_name == "casme_026_0101":
                        face_right = face_right + 2
                    face = image[face_top:face_bottom, face_left:face_right]  # 裁剪人脸区域
                    # 不调整尺寸
                    # face = cv2.resize(face, (128, 128))  # 调整尺寸为 128x128
                    # 保存图片
                    cv2.imwrite(os.path.join(dir_crop_sub_vid, "img{}.jpg").format(img_name), face)
                    tq.update()  # 更新进度


if __name__ == "__main__":
    with open("/kaggle/working/PreSolve/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    crop_images(opt)
