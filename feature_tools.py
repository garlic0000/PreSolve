import math
import numpy as np
import cv2


def get_top_optical_flows(optflows, percent):
    """
    选择占比最高的光流向量
    筛选出最显著的光流
    如果是三维的 则展开为二维
    如果光流向量已经归一化（即每个向量的模长都被调整为相同的值，比如 1），
    那么排序就没有意义，因为所有光流向量的模长都是一样的，无法区分大小，也就无法根据模长选出“最显著”的光流
    """
    assert type(optflows) == np.ndarray, "optflows must be numpy ndarray"
    tmp_optflows = optflows
    # 如果输入是三维 (height, width, 2)，则展平为 (height*width, 2)
    if optflows.ndim == 3 and optflows.shape[-1] == 2:
        tmp_optflows = optflows.reshape(-1, 2)
    # 已经是二维 不需要任何改变
    elif optflows.ndim == 2 and optflows.shape[-1] == 2:
        tmp_optflows = optflows
    else:
        raise "shape of optflows is invalid"

    length = len(optflows)
    top_n = int(length * percent)  # 从中选取幅度最大的前 top_n 个光流向量
    # np.linalg.norm 计算每个光流向量的幅度（即 sqrt(dx^2 + dy^2)）
    # np.argsort 根据光流向量的幅度进行升序排序，返回排序后的索引
    new_indices = np.argsort(np.linalg.norm(tmp_optflows, axis=-1))
    # 通过排序后的索引，选取幅度最大的 top_n 个光流向量
    ret_optflows = tmp_optflows[new_indices][length - top_n:]
    return ret_optflows


def get_rectangle_roi_boundary(indices, landmarks,
                               horizontal_bound=0, vertical_bound=0):
    """ calculate a boundary of a roi that consists of a bunch of landmarks

    Args:
        indices: indices of landmarks, must be tuple, list of numpy.dnarray
    Returns:
        left_bound: left boundary of the rectangle roi
        top_bound: top boundary of the rectangle roi
        right_bound: right boundary of the rectangle roi
        bottom_bound: bottom boundary of the rectangle roi
    """
    # 获取兴趣区域的边界
    assert type(horizontal_bound) == int, "horizontal_bound must be integer"
    assert type(vertical_bound) == int, "vertical_bound must be integer"
    if type(indices) == tuple or type(indices) == list:
        indices = np.array(indices)
    elif type(indices) == np.ndarray:
        pass
    else:
        raise "type of indices is incorrect"

    roi_landmarks = landmarks[indices]  # 根据indices获取对应位置的坐标
    # 获取坐标x的最大最小, 坐标y的最大最小
    # axis 表示沿着行的方向 比较每一列的最大值和最小值  roi_landmarks为n行2列
    left_bound, top_bound = np.min(roi_landmarks, axis=0)
    right_bound, bottom_bound = np.max(roi_landmarks, axis=0)
    # 上面两行代码没问题
    # 但是 +-horizontal_bound  +-vertical_bound之后有几个有问题
    # 会出现nose_roi_left > nose_roi_right问题
    # casme_030_0505 中有17张图片计算出horizontal_bound为负数的情况
    return left_bound - horizontal_bound, top_bound - vertical_bound, \
           right_bound + horizontal_bound, bottom_bound + vertical_bound


def get_rois(mat, landmarks, indices, horizontal_bound=3, vertical_bound=3):
    """ get rois with indeices of landmarks

    Args:
        mat: a rgb image or flow image
        landmarks: landmarks of face region
        indeices: indeices of landmarks
        horizontal_bound:
        vertical_bound:
    Returns:
        a ndarray of roi mat
    """
    # 取光流 的区域
    # 获取兴趣区域 矩形
    # indices的类型为元组或者数组 转化成 numpy数组
    if type(indices) == tuple or type(indices) == list:
        indices = np.array(indices)
    elif type(indices) == np.ndarray:
        pass
    else:
        raise "type of indices is incorrect"

    assert type(landmarks) == np.ndarray, "landmarks should be numpy.ndarray"

    roi_list = []
    for landmark in landmarks[indices]:
        # 从传入的indices遍历关键点坐标
        x = landmark[0].item()  # 横坐标
        y = landmark[1].item()  # 纵坐标
        # 是否会超过mat的范围？
        roi_list.append(mat[y - vertical_bound: y + vertical_bound + 1,
                        x - horizontal_bound: x + horizontal_bound + 1, :])
    return np.stack(roi_list, axis=0)


def optflow_normalize(flow):
    """
    归一化光流向量，生成特征向量
    normalize optical flows

    Args:
        flow: np.ndarry, shape of flow should be (-1, 2)

    Returns:
        a np.ndarray, the shape of return is (2,)
    """
    # 原项目中归一化
    # 这个可能要使用原始光流大小
    # 因为需要计算光流的幅度大小
    # 而且这个函数进行了光流的归一化 可能在传入光流之前不需要进行归一化
    assert flow.dtype == np.float32, (
        "element type of optflow should be float32")

    delta = 0.000001
    # 这个求和不管正负号
    # 将 flow 在第 0 维（即所有点的光流向量）进行逐元素求和。
    # 结果是一个形状为 (2,) 的向量，代表光流在 x 和 y 方向上的总和。
    # 所有的x分量相加 所有的y分量相加
    # 最终sum_flow会变成sum_flow = [x分量之和, y分量之和]
    # 直接求 x 和 y 分量的和，可以简化所有光流向量为一个全局的方向向量，用于描述整体的运动趋势。
    # 这种方法在处理大量光流时非常有用，因为它能够保留主要的运动信息
    sum_flow = np.sum(flow, axis=0)
    # 各向量平方和的平方根
    flow_one = sum_flow / (np.linalg.norm(sum_flow) + delta)
    # flow.shape[0]为光流的个数
    # 每个光流的x的平方和y的平方开方之后求和再除以总光流个数
    # 求模的平均值
    # 展平光流之后的平均模值求法 (flow, axis=1) 二维展平光流 (flow, axis=2) 三维原始光流
    # 这两中方法本质上相同
    average_module = np.sum(np.linalg.norm(flow, axis=1)) / flow.shape[0]
    feature = flow_one * average_module
    return feature



def get_main_direction_flow(array_flow, direction_region):
    """
    提取主方向的光流向量
    get all the flow vectors that are main directional in a region of flow

    Args:
        array_flow: a ndarray of flows
    Returns:
        a ndarray of flows that are main directional in a region of flow
    """

    # 这里是不是要使用原始光流？
    # 这个可以使用归一化光流 因为只使用了光流的方向
    # 使用两种光流 一种处理过的 另一种未处理过的用于提取方向
    # 将光流矩阵展平并计算其角度
    # 光流数据一般是三维的
    array_flow = array_flow.reshape(-1, 2)
    # 每个像素的光流可以用一个二维向量表示，包含水平（x方向）和垂直（y方向）两个分量
    # cv2.cartToPolar将x和y分量分别转换为极坐标下的幅度和角度
    # - 运动的向量的大小 运动的速度或强度 在这段代码中没有使用它，所以用下划线表示忽略
    # angs 运动的方向
    _, angs = cv2.cartToPolar(array_flow[..., 0], array_flow[..., 1])
    # 为每个方向区间初始化一个列表
    direction_flows = [[] for i in range(len(direction_region))]

    # 遍历每个角度，按方向区间分类光流
    # 将光流按照方向分组。每个光流向量通过角度计算确定属于哪个方向区间，然后将光流归类到相应的方向区间
    for i, ang in enumerate(angs):
        for index, direction in enumerate(direction_region):
            if len(direction) == 2:  # 判断该方向区间是否由两个边界值组成
                # 这意味着该区间是一个普通的角度范围，比如 [π/6, π/2]，表示某个连续的角度范围
                if ang >= direction[0] and ang < direction[1]:  # 如果角度在这个方向区间内
                    # 说明这是一个跨越 0 或 2π 的方向区间，形如 [7π/4, 2π, 0, π/4]。
                    # 这种情况下，光流角度可能在两个不同的区间之间，例如从 7π/4 到 2π 再到 0 到 π/4。
                    direction_flows[index].append(array_flow[i])
                    break  # 找到合适的方向区间后，跳出内层循环
            elif len(direction) == 4:  # 判断该方向区间是否有四个边界值（处理跨越0的情况）
                if (ang >= direction[0]
                        or (ang >= direction[2] and ang < direction[3])):
                    direction_flows[index].append(array_flow[i])
                    break

    # 找到包含最多光流向量的方向
    # np.argmax 返回数组中最大值的索引
    max_count_index = np.argmax(
        # 计算每个方向的光流个数
        np.array([len(x) for x in direction_flows])).item()

    # 最终的返回值是一个二维数组，表示包含最多光流向量的方向区间下的所有光流向量
    # 每一行是一个光流向量，形如 [dx, dy]，其中 dx 和 dy 是该光流向量的水平和垂直分量
    return np.stack(direction_flows[max_count_index], axis=0)


def cal_global_optflow_vector(flows, landmarks):
    """
    计算面部鼻子区域的光流向量，作为指示头部运动的全局光流
    calculates optical flow vector of nose region

    calculates array of optical flows of nose region as the global optical flow
    to indicate head motion, and then calculates the normalized vector of the
    array.

    Args:
        flows: flows of a image
        landmarks: landmarks of the face region
    Returns:
        global optical flow vector.
    """

    # 这个函数没有任何处理？
    # 使用下面这个函数的处理？
    # python函数内嵌套函数？
    def _cal_partial_opt_flow(indices, horizontal_bound, vertical_bound):
        """
        计算指定关键点区域的光流
        """

        (nose_roi_left, nose_roi_top, nose_roi_right,
         nose_roi_bottom) = get_rectangle_roi_boundary(
            indices, landmarks,
            horizontal_bound, vertical_bound)
        """
        flow_nose_roi is empty after extraction, checking boundaries...
ROI boundaries: top=139, bottom=153, left=34, right=27
        """
        # 确保左右边界正确
        if nose_roi_left > nose_roi_right:
            print("nose_roi_left > nose_roi_right")
            nose_roi_left, nose_roi_right = nose_roi_right, nose_roi_left  # 交换左右边界

        # 确保上下边界正确
        if nose_roi_top > nose_roi_bottom:
            print("nose_roi_top > nose_roi_bottom")
            nose_roi_top, nose_roi_bottom = nose_roi_bottom, nose_roi_top  # 交换上下边界

        # 使用np.max和np.min确保ROI边界不越界
        nose_roi_left = np.max([nose_roi_left, 0])
        nose_roi_top = np.max([nose_roi_top, 0])
        nose_roi_right = np.min([nose_roi_right, flows.shape[1] - 1])
        nose_roi_bottom = np.min([nose_roi_bottom, flows.shape[0] - 1])
        # 根据修正后的边界提取ROI
        flow_nose_roi = flows[nose_roi_top:nose_roi_bottom + 1, nose_roi_left:nose_roi_right + 1]
        flow_nose_roi = flow_nose_roi.reshape(-1, 2)
        return flow_nose_roi

    # LEFT_EYE_CORNER_INDEX = 36  # 左眼外眼角
    # RIGHT_EYE_CORNER_INDEX = 45  # 右眼外眼角
    LEFT_EYE_CONER_INDEX = 39  # 原项目数据 左眼内眼角
    RIGHT_EYE_CONER_INDEX = 42  # 原项目数据 右眼内眼角
    left_eye_coner = landmarks[LEFT_EYE_CONER_INDEX]
    right_eye_coner = landmarks[RIGHT_EYE_CONER_INDEX]
    # casme_030_0505 中有17张图片计算出负数
    length_between_coners = (right_eye_coner[0] - left_eye_coner[0]) / 2

    """
    如果您使用的是自定义的关键点模型，建议确认索引 29 和 30 对应的位置，以确保代码的正确性。
    0-16: 下颌线（Jawline）
    17-21: 左眉毛（Left Eyebrow）
    22-26: 右眉毛（Right Eyebrow）
    27-30: 鼻梁（Nose Bridge）
    30-35: 鼻尖及下部鼻子（Lower Nose）
    36-41: 左眼（Left Eye）
    42-47: 右眼（Right Eye）
    48-59: 外嘴唇（Outer Lip）
    60-67: 内嘴唇（Inner Lip）
    """
    """
    27: 鼻梁的起点，位于眉毛之间。
    28: 鼻梁中间的点。
    29: 鼻梁接近鼻尖的点。
    30: 鼻尖（Nose Tip）。
    
    鼻尖及其附近的区域在面部表情变化中相对稳定，减少了由于表情引起的关键点移动带来的干扰
    鼻尖是面部的中心点，选择其周围的区域可以更好地捕捉到面部的整体运动模式，尤其是在处理面部表情或头部姿态变化时。
    """
    flow_nose_roi_list = []
    flow_nose_roi_list.append(
        _cal_partial_opt_flow(
            # 选择 29 30 这两个 31不包括在内
            np.arange(29, 30 + 1),
            horizontal_bound=int(length_between_coners * 0.35),
            vertical_bound=int(length_between_coners * 0.35)))
    flow_nose_roi = np.stack(flow_nose_roi_list).reshape(-1, 2)
    if flow_nose_roi.size == 0:
        raise ValueError("flow_nose_roi is empty, check ROI boundaries or flow data.")
    flow_nose_roi = get_main_direction_flow(
        flow_nose_roi,
        direction_region=[
            (1 * math.pi / 4, 3 * math.pi / 4),  # 代表从 45 度到 135 度的方向
            (3 * math.pi / 4, 5 * math.pi / 4),  # 代表从 135 度到 225 度的方向
            (5 * math.pi / 4, 7 * math.pi / 4),  # 代表从 225 度到 315 度的方向
            (7 * math.pi / 4, 8 * math.pi / 4, 0, 1 * math.pi / 4),  # 代表从 315 度到 45 度，处理跨越 0 度的情况
        ])
    if flow_nose_roi is None:
        raise ValueError("get_main_direction_flow returned None, check flow calculation.")
    # 按光流向量的幅度（即长度）从大到小排序后，选择前 88% 的向量
    flow_nose_roi = get_top_optical_flows(flow_nose_roi, percent=0.88)
    # 这里的归一化可能要注释掉
    # 在这里进行归一化之后 可能后面就没有了可比性
    # 比如 roi_flows_adjust = roi_flows - global_optflow_vector
    # global_flow_vector = optflow_normalize(flow_nose_roi)
    # 直接使用 最后统一归一化
    # operands could not be broadcast together with shapes (12,11,11,2) (813,2)
    # global_flow_vector = flow_nose_roi.reshape(-1, 2)
    global_flow_vector = optflow_normalize(flow_nose_roi)
    return global_flow_vector


def calculate_roi_freature_list(flow, landmarks, radius):
    """
    radius怎么确定？
    """
    assert flow.dtype == np.float32, (
        "element type of optflow should be float32")
    # 使用原始光流 暂时进行注释
    assert np.max(flow) <= 1, "max value shoued be less than 1"

    roi_flows = get_rois(
        flow, landmarks,
        indices=[
            18, 19, 20,  # 左眉毛
            23, 24, 25,  # 右眉毛
            28, 30,  # 鼻子
            48, 51, 54, 57  # 嘴巴
        ],
        horizontal_bound=radius,  # 关键点周围提取的区域大小
        vertical_bound=radius
    )

    # 计算全局光流向量
    global_optflow_vector = cal_global_optflow_vector(flow, landmarks)
    # 通过从提取的 ROI 光流中减去全局光流向量，来调整局部光流数据，以消除全局运动的影响
    roi_flows_adjust = roi_flows - global_optflow_vector
    print("sdfsd")
    print(roi_flows.shape)
    print("dddd")
    print(global_optflow_vector.shape)
    roi_feature_list = []  # feature in face
    for roi_flow in roi_flows_adjust:
        roi_main_direction_flow = get_main_direction_flow(
            roi_flow,
            # 四个方向区域，以弧度为单位
            direction_region=[
                (1 * math.pi / 6, 5 * math.pi / 6),
                (5 * math.pi / 6, 7 * math.pi / 6),
                (7 * math.pi / 6, 11 * math.pi / 6),
                (11 * math.pi / 6, 12 * math.pi / 6, 0, 1 * math.pi / 6),
            ])
        # 从提取的主要方向光流中选取前 60% 的光流，以减少噪声或不重要的信息
        roi_main_direction_flow = get_top_optical_flows(
            roi_main_direction_flow, percent=0.6)
        # 对选取的光流特征进行归一化处理
        roi_feature = optflow_normalize(roi_main_direction_flow)
        # 测试
        # roi_feature = optflow_normalize_(roi_feature)
        roi_feature_list.append(roi_feature)
    return np.stack(roi_feature_list, axis=0)
