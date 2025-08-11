import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import glob
import os

# 调节阈值
threshold_constant = 1.3

# 每一帧图像的时间间隔
time_interval = 0.1

# 文件夹路径
image_folder = "C:\\Users\\PC\\Desktop\\final year project\\pictures\\20250119"

# 存储所有图片的距离
output_distances = []

# 加载文件夹中的所有图片（假设文件夹中只有图片文件）
image_files = glob.glob(os.path.join(image_folder, "*.png"))

#选择需要的图片
image_files=image_files[:50]

# 检查是否有图片
if not image_files:
    raise ValueError("No images found in the folder.")

# 全局变量用于存储 ROI
rois = []

# 回调函数，用于存储 ROI 的坐标
def on_select(eclick, erelease):
    """
    回调函数，用于存储矩形区域的坐标。
    eclick 和 erelease 是鼠标按下和释放的事件。
    """
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # 左上角
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # 右下角
    rois.append((min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)))  # 存储为 (y_min, y_max, x_min, x_max)
    print(f"ROI selected: {rois[-1]}")

# 交互式矩形选择器
def select_rois(image_array):
    """
    使用 RectangleSelector 绘制并选择两个 ROI。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_array, cmap='gray')
    ax.set_title("Draw two rectangles for the ROIs")
    rectangle_selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],  # 左键
        interactive=True
    )
    plt.show(block=True)

# Step 1: 在第一张图片上选择 ROI
print("Please draw two rectangles on the first image to select the ROIs.")
first_image_path = image_files[0]
first_image = Image.open(first_image_path).convert("L")  # 转换为灰度图像
first_image_array = np.array(first_image)
select_rois(first_image_array)

# 检查是否选择了两个 ROI
if len(rois) < 2:
    raise ValueError("Please select exactly two ROIs.")

# 提取两个 ROI 的坐标
roi1 = rois[0]
roi2 = rois[1]

# Step 2: 对文件夹中的所有图片进行处理
for image_path in image_files:
    # 加载图片并转换为灰度图像
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)

    # 提取两个 ROI
    roi1_array = image_array[roi1[0]:roi1[1], roi1[2]:roi1[3]]
    roi2_array = image_array[roi2[0]:roi2[1], roi2[2]:roi2[3]]

    # Step 3: 对 ROI 进行二值化处理
    threshold1 = np.mean(roi1_array) + threshold_constant*np.std(roi1_array)  # 自适应阈值
    threshold2 = np.mean(roi2_array) + threshold_constant*np.std(roi2_array)  # 自适应阈值
    roi1_binary = roi1_array > threshold1
    roi2_binary = roi2_array > threshold2

    # Step 4: 计算质心
    centroid1 = center_of_mass(roi1_binary)  # ROI 1 的质心
    centroid2 = center_of_mass(roi2_binary)  # ROI 2 的质心

    # 将质心坐标转换为全局坐标
    centroid1_global = (centroid1[0] + roi1[0], centroid1[1] + roi1[2])
    centroid2_global = (centroid2[0] + roi2[0], centroid2[1] + roi2[2])

    # Step 5: 计算两个质心之间的距离
    distance = np.sqrt((centroid2_global[0] - centroid1_global[0])**2 +
                       (centroid2_global[1] - centroid1_global[1])**2)

    # 将距离保存到数组中
    output_distances.append(distance)

# 输出图像
output_distances = np.array(output_distances)
numbers = np.array(np.linspace(0, time_interval*(len(image_files)-1), len(image_files)))
plt.plot(numbers,output_distances,label='drifting bead')
plt.xlabel('time')
plt.ylabel('displacement/pixel')
plt.legend()
plt.show()