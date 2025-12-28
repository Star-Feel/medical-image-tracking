import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def read_video_frames(video_path, frame_indices, target_size=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    for idx in frame_indices:
        if idx < 0 or idx >= total_frames:
            raise ValueError(f"视频 {video_path} 的帧索引 {idx} 超出范围（总帧数：{total_frames}）")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"读取视频 {video_path} 的第 {idx} 帧失败")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        frames.append(img)
    
    cap.release()
    return frames

def add_title_to_row(row_img, title_text, font_size=40, padding=10, title=False):
    row_width, row_height = row_img.size
    if title:
        total_height = row_height + font_size + padding
    else:
        total_height = row_height

    new_img = Image.new("RGB", (row_width, total_height), color="white")
    draw = ImageDraw.Draw(new_img)
    
    if title == True:
        font = None
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
        ]
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        if font is None:
            font = ImageFont.load_default(size=font_size)

        text_bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (row_width - text_width) // 2
        text_y = padding // 2

        draw.text((text_x, text_y), title_text, font=font, fill="black")

        new_img.paste(row_img, (0, font_size + padding))
    else:
        new_img.paste(row_img, (0, 0))
    return new_img

def main():
    VIDEO_PATHS = [
        "/root/projects/lite-tracker/results/1_grid20.mp4",
        "/root/projects/lite-tracker/results/2_grid20.mp4",
        "/root/projects/lite-tracker/results/3_grid20.mp4",
        "/root/projects/lite-tracker/results/SuPer.mp4"
    ]
    FRAME_INDICES_PER_VIDEO = [
        [100, 110, 120, 130, 140, 150],
        [70, 80, 90, 100, 110, 120],
        [30, 40, 50, 60, 70, 80],
        [90, 110, 130, 150, 170, 190],
    ]
    TARGET_FRAME_SIZE = (400, 300)
    OUTPUT_IMAGE_PATH = "final_collage.png"
    TITLE_FONT_SIZE = 40

    if len(VIDEO_PATHS) != 4:
        raise ValueError("必须提供4个视频路径")

    titled_rows = []

    for i, video_path in enumerate(VIDEO_PATHS):
        print(f"正在处理视频: {video_path}")
        # 读取6帧
        frames = read_video_frames(video_path, FRAME_INDICES_PER_VIDEO[i], TARGET_FRAME_SIZE)
        frame_width, frame_height = frames[0].size
        row_width = frame_width * 6
        row_height = frame_height
        row_img = Image.new("RGB", (row_width, row_height))
        for j, frame in enumerate(frames):
            row_img.paste(frame, (j * frame_width, 0))
        
        if i < 3:
            title = "STIR"
        else:
            title = "SuPer"
        titled_row = add_title_to_row(row_img, title, font_size=TITLE_FONT_SIZE, title=(i==0 or i==3))
        titled_rows.append(titled_row)

    collage_width = titled_rows[0].size[0]
    collage_height = sum([row.size[1] for row in titled_rows])
    final_collage = Image.new("RGB", (collage_width, collage_height))
    current_y = 0
    for row in titled_rows:
        final_collage.paste(row, (0, current_y))
        current_y += row.size[1]

    final_collage.save(OUTPUT_IMAGE_PATH)
    print(f"save at: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行出错: {e}")