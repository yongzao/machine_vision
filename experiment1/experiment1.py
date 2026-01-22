from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# RGB转灰度图
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# 自定义2D卷积函数
def convolve2d(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2  # 3×3核填充1个像素
    padded_img = np.pad(image, pad_width=pad, mode='constant', constant_values=0)   # 填充保证卷积前后尺寸一致
    H, W = image.shape
    output = np.zeros_like(image, dtype=np.float32)  # 浮点型避免计算溢出
    # 滑动窗口
    for i in range(H):
        for j in range(W):
            #加权求和
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(window * kernel)
    # 截断到像素合法范围并转回uint8
    return np.clip(output, 0, 255).astype(np.uint8)

# 计算单通道颜色直方图
def compute_color_hist(channel):
    hist = np.zeros(256, dtype=int)
    for pixel in channel.flatten():
        hist[pixel] += 1
    return hist

# 提取灰度共生矩阵（GLCM）纹理特征
def compute_glcm_features(gray_img, quant_level=16):
    # 灰度量化
    gray_quant = (gray_img // (256 // quant_level)).astype(np.uint8)
    H, W = gray_quant.shape
    glcm = np.zeros((quant_level, quant_level), dtype=np.float32)
    # 计算水平方向共生矩阵
    for i in range(H):
        for j in range(W - 1):
            g1 = gray_quant[i, j]
            g2 = gray_quant[i, j + 1]
            glcm[g1, g2] += 1
    # 归一化
    total = glcm.sum()
    if total > 0:
        glcm = glcm / total
    # 计算纹理特征
    i, j = np.indices(glcm.shape)
    energy = np.sum(glcm ** 2)  # 能量（角二阶矩）
    contrast = np.sum(((i - j) ** 2) * glcm)  # 对比度
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # 熵（加小值避免log(0)）
    # 相关性
    mu_x = np.sum(i * glcm)
    mu_y = np.sum(j * glcm)
    sigma_x = np.sqrt(np.sum(((i - mu_x) ** 2) * glcm))
    sigma_y = np.sqrt(np.sum(((j - mu_y) ** 2) * glcm))
    correlation = np.sum((i - mu_x) * (j - mu_y) * glcm) / (sigma_x * sigma_y) if (sigma_x * sigma_y) != 0 else 0.0
    return np.array([energy, contrast, entropy, correlation])


# 图像读取与预处理
img_path = "Lena.jpg"
img = Image.open(img_path).convert("RGB")  # 确保为RGB格式
img_np = np.array(img)
gray_img = rgb2gray(img_np)  # 转为灰度图用于卷积

# 定义卷积核并执行滤波
# 给定卷积核（X+Y方向）
custom_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
custom_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
custom_filter_x = convolve2d(gray_img, custom_kernel_x)
custom_filter_y = convolve2d(gray_img, custom_kernel_y)
custom_filter_merged = np.sqrt(custom_filter_x.astype(float) ** 2 + custom_filter_y.astype(float) ** 2)
custom_filter_merged = np.clip(custom_filter_merged, 0, 255).astype(np.uint8)
# Sobel算子（X+Y方向）
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
sobel_filter_x = convolve2d(gray_img, sobel_kernel_x)
sobel_filter_y = convolve2d(gray_img, sobel_kernel_y)
sobel_filter_merged = np.sqrt(sobel_filter_x.astype(float) ** 2 + sobel_filter_y.astype(float) ** 2)
sobel_filter_merged = np.clip(sobel_filter_merged, 0, 255).astype(np.uint8)

# 计算颜色直方图
r_channel = img_np[:, :, 0]
g_channel = img_np[:, :, 1]
b_channel = img_np[:, :, 2]
hist_r = compute_color_hist(r_channel)
hist_g = compute_color_hist(g_channel)
hist_b = compute_color_hist(b_channel)

# 提取并保存纹理特征
texture_features = compute_glcm_features(gray_img)
np.save("texture_features.npy", texture_features)
print(f"纹理特征已保存至 texture_features.npy，特征值：{texture_features}")
plt.figure(figsize=(18, 6))
# 给定卷积核滤波结果
plt.subplot(1, 3, 1)
plt.imshow(custom_filter_merged, cmap="gray")
plt.title("给定卷积核滤波", fontsize=12)
plt.axis("off")
# Sobel算子滤波结果
plt.subplot(1, 3, 2)
plt.imshow(sobel_filter_merged, cmap="gray")
plt.title("Sobel算子滤波", fontsize=12)
plt.axis("off")
# 颜色直方图
plt.subplot(1, 3, 3)
plt.plot(hist_r, color="red", label="Red Channel")
plt.plot(hist_g, color="green", label="Green Channel")
plt.plot(hist_b, color="blue", label="Blue Channel")
plt.title("颜色直方图", fontsize=12)
plt.xlabel("Pixel Value (0-255)")
plt.ylabel("Pixel Count")
plt.xlim(0, 255)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()