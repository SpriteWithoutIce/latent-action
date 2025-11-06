import json
import os

# JSON 文件路径
json_path = "/home/linyihan/linyh/latent-action/latent_action_visualizations/metadata.json"

# 图片所在目录（可改为绝对路径）
image_dir = "/home/linyihan/linyh/latent-action/latent_action_visualizations"

# 读取 JSON
with open(json_path, "r") as f:
    data = json.load(f)

# 保留 saved_images >= 9 的文件名
keep_files = {item["filename"] for item in data if item["saved_images"] >= 9}

# 遍历目录，删除不在 keep_files 列表中的图片
for file in os.listdir(image_dir):
    if file.endswith(".png") and file not in keep_files:
        file_path = os.path.join(image_dir, file)
        print(f"Deleting {file_path}")
        os.remove(file_path)
