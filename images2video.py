import cv2
import os
from tqdm import tqdm

# 配置参数
images_dir = "/root/projects/lite-tracker/data/SuPerDataset/img"  
video_path = "./data/SuPer.mp4"                                   
fps = 30.0                                                       
resize_to_same = True                                             

image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
image_files = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(image_extensions)
]

image_files.sort()

if not image_files:
    print(f"错误：在 {images_dir} 中未找到图片文件！")
    exit(1)

os.makedirs(os.path.dirname(video_path), exist_ok=True)

first_image_path = os.path.join(images_dir, image_files[0])
first_image = cv2.imread(first_image_path)
if first_image is None:
    print(f"错误：无法读取第一张图片 {first_image_path}！")
    exit(1)
height, width = first_image.shape[:2]
print(f"视频尺寸：{width}x{height}，帧率：{fps}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    video_path,
    fourcc,
    fps,
    (width, height)
)

for file in tqdm(image_files):
    image_path = os.path.join(images_dir, file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"警告：跳过无法读取的图片 {image_path}")
        continue
    
    if resize_to_same and (image.shape[0] != height or image.shape[1] != width):
        image = cv2.resize(image, (width, height))
        print(f"警告：图片 {file} 尺寸不匹配，已调整为 {width}x{height}")

    video_writer.write(image)

video_writer.release()
cv2.destroyAllWindows()

print(f"\n视频合成完成！保存路径：{video_path}")
print(f"总帧数：{len(image_files)}，视频时长：{len(image_files)/fps:.2f} 秒")