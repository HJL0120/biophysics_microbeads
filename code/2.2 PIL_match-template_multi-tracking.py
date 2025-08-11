import numpy as np
from PIL import Image
import cv2
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# 时间间隔
time_interval = 0.1

# 文件夹路径
image_folder = "C:\\Users\\PC\\Desktop\\final year project\\pictures\\20250119"

# 存储所有图片的距离
output_distances = []  

# 加载文件夹中的所有图片（假设文件夹中只有图片文件）
image_files = glob.glob(os.path.join(image_folder, "*.png"))
image_files = image_files[:50]

# 检查是否有图片
if not image_files:
    raise ValueError("No images found in the folder.")

# 存储选中的点坐标
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
    ax.set_title("Draw over two rectangles for the ROIs")
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

# 检查是否选择了至少两个 ROI
if len(rois) < 2:
    raise ValueError("Please select at least two ROIs.")

# Step 2: 对每个 ROI 进行模板匹配
dynamics_global_coordinates=[]
# 提取第一个 ROI 作为模板
template_roi = rois[0]

for image_path in image_files:
    image = Image.open(image_path).convert("L")  # 转换为灰度图像
    image_array = np.array(image)
    template = image_array[template_roi[0]:template_roi[1], template_roi[2]:template_roi[3]]
    global_coordinates = []  # 存储每个 ROI 的最佳匹配位置的全局坐标
    for i, roi in enumerate(rois[1:]):  # 从第二个 ROI 开始
        # 提取当前 ROI
        image_roi = image_array[roi[0]:roi[1], roi[2]:roi[3]]
        
        # 检查 ROI 是否比模板小
        if image_roi.shape[0] < template.shape[0] or image_roi.shape[1] < template.shape[1]:
            print(f"ROI {i + 1} is smaller than the template. Skipping this ROI.")
            continue  # 跳过这个 ROI
        
        # 使用 OpenCV 的 matchTemplate 进行模板匹配
        result = cv2.matchTemplate(image_roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 最佳匹配位置
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        
        # 将匹配位置转换到整体坐标系
        top_left_global = (top_left[0] + roi[2], top_left[1] + roi[0])
        bottom_right_global = (bottom_right[0] + roi[2], bottom_right[1] + roi[0])
        
        # 计算质心
        centroid = ((top_left_global[0] + bottom_right_global[0]) / 2, 
                    (top_left_global[1] + bottom_right_global[1]) / 2)
        
        # 将全局坐标保存到列表中
        global_coordinates.append(centroid)
    dynamics_global_coordinates.append(global_coordinates)

#选择研究的微珠运动方向
direction='y'
#选择微珠序号(注意0号为第一个微珠)
bead_index=1
def switch(case):
    cases = {
    'x': 0,
    'y': 1
    }
    return cases.get(case, 'default Case')
direction = switch(direction)
one_bead_axis=np.array(dynamics_global_coordinates).T[direction][bead_index]
# 创建时间轴
time_points = np.linspace(0, time_interval * (len(image_files) - 1), len(image_files))
plt.plot(time_points,one_bead_axis,label='one direction drifting')
plt.xlabel('time')
plt.ylabel('displacement/pixel')
plt.legend()
plt.show()