import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
def detect_lane_lines(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    original_img = img.copy()
    height, width = img.shape[:2]
    # 图像预处理
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Canny边缘检测
    canny = cv2.Canny(blur, 40, 120)
    # 定义感兴趣区域（仅保留道路区域，过滤无关背景）
    def region_of_interest(edge_img):
        mask = np.zeros_like(edge_img)
        # 构建梯形感兴趣区域（适配道路“近宽远窄”的透视）
        vertices = np.array([[
            (width*0.05, height),         # 左下（更靠左）
            (width*0.15, height*0.25),    # 左上（更高、更靠左）
            (width*0.85, height*0.25),    # 右上（更高、更靠右）
            (width*0.95, height)          # 右下（更靠右）
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_edge = cv2.bitwise_and(edge_img, mask)

        return masked_edge
    masked_canny = region_of_interest(canny)
    plt.imshow(masked_canny, cmap='gray')
    plt.title("ROI后的边缘图像")
    plt.axis('off')
    plt.show()
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        masked_canny,
        rho=1,               # 霍夫变换检测直线
        theta=np.pi/180,     # 极坐标角度步长（弧度）
        threshold=45,        # 直线的最小投票数
        minLineLength=100,   # 检测的最小线段长度
        maxLineGap=25        # 线段间的最大间隙（小于则合并）
    )
    # 筛选左右车道线
    left_lines = []  # 左车道线（斜率：-1.8 ~ -0.8）
    right_lines = [] # 右车道线（斜率：0.6 ~ 2）
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算斜率
            if x2 - x1 == 0:
                right_lines.append(line[0])
            slope = (y2 - y1) / (x2 - x1)
            # 过滤路边、护栏等干扰线
            if 0.6 < slope < 1.8:
                right_lines.append(line[0])
            elif -1.8 < slope < -0.6:
                left_lines.append(line[0])
    # 拟合左右车道线（计算平均线段）
    def draw_lane_lines(img, lines):
        if not lines:
            return img
        # 计算所有线段的斜率和截距的平均值
        slope_sum = 0
        intercept_sum = 0
        count = 0
        for x1, y1, x2, y2 in lines:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slope_sum += slope
            intercept_sum += intercept
            count += 1
        if count == 0:
            return img
        avg_slope = slope_sum / count
        avg_intercept = intercept_sum / count
        # 确定车道线的上下端点（覆盖完整感兴趣区域）
        y_bottom = height
        y_top = int(height * 0.65)  # 与ROI顶部对齐
        x_bottom = int((y_bottom - avg_intercept) / avg_slope)
        x_top = int((y_top - avg_intercept) / avg_slope)
        # 绘制车道线（调整线宽适配宽道路）
        cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), (0, 255, 0), 6)
        return img
    # 绘制左右车道线
    result_img = original_img.copy()
    result_img = draw_lane_lines(result_img, left_lines)
    result_img = draw_lane_lines(result_img, right_lines)
    return original_img, result_img
img_path = "img_3.png"
original_rgb, result_rgb = detect_lane_lines(img_path)
# 显示结果
plt.figure(figsize=(16, 8))
# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("原始图像")
plt.axis("off")
# 道线检测结果
plt.subplot(1, 2, 2)
plt.imshow(result_rgb)
plt.title("道线检测结果")
plt.axis("off")
plt.tight_layout()
plt.show()