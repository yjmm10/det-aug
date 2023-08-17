# from PIL import Image

# def gif_to_png(gif_file, png_dir):
#     # 打开GIF文件
#     with Image.open(gif_file) as im:
#         # 遍历每一帧
#         for i in range(im.n_frames):
#             # 选择当前帧
#             im.seek(i)
#             # 将当前帧转换为RGB格式
#             rgb_im = im.convert('RGB')
#             # 生成PNG文件名
#             png_file = f"{png_dir}/{os.path.splitext(os.path.basename(gif_file))[0]}_{i:03d}.png"
#             # 保存当前帧为PNG文件
#             rgb_im.save(png_file)
#             break

# import os

# def batch_gif_to_png(gif_dir, png_dir):
#     # 遍历GIF目录
#     for filename in os.listdir(gif_dir):
#         # 判断是否为GIF文件
#         if filename.endswith(".gif"):
#             # 拼接完整文件路径
#             gif_file = os.path.join(gif_dir, filename)
#             print(gif_file)
#             # 生成PNG目录
#             os.makedirs(png_dir, exist_ok=True)
#             # 调用gif_to_png函数进行转换
#             gif_to_png(gif_file, png_dir)

# # 示例用法
# gif_dir = "F:\Project\DA-GUI\dataset\yhk"
# png_dir = "F:\Project\DA-GUI\dataset\yhk_output"
# batch_gif_to_png(gif_dir, png_dir)




import os


# # # 文件夹下重新命名
# folder = r"F:\Project\DA-GUI\dataset\yyzz_aug\label"
# new_prefix = "aug0_"
# start_index = 1

# for i, filename in enumerate(os.listdir(folder)):
#     fn,file_extension = os.path.splitext(filename)
#     # new_filename = f"{new_prefix}{i+start_index:05d}{file_extension}"
#     # 加前缀
#     new_filename = f"{new_prefix}{fn}{file_extension}"
#     old_file_path = os.path.join(folder, filename)
#     new_file_path = os.path.join(folder, new_filename)
#     os.rename(old_file_path, new_file_path)


# folder = "F:\Project\DA-GUI\dataset\sfz"
# old_string = "反面"
# new_string = "1"
# # file_extension = ".txt"

# for filename in os.listdir(folder):
#     new_filename = filename.replace(old_string, new_string)
#     old_file_path = os.path.join(folder, filename)
#     new_file_path = os.path.join(folder, new_filename)
#     os.rename(old_file_path, new_file_path)



# 缩放图片
import cv2
import os

# 读取图像
img = cv2.imread("F:\Project\DA-GUI\dataset\hh\idcard_20230131152033_1_0.png")

# 缩放目录下的图片，保持图片名不变
def resize_images(input_dir, output_dir,scale=1/5):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 循环遍历输入目录中的所有图像文件
    for filename in os.listdir(input_dir):
        # 检查文件扩展名是否为图像
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 构建输入和输出文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 加载图像并调整大小
            img = cv2.imread(input_path)
            resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # 保存缩放后的图像
            cv2.imwrite(output_path, resized_img)

            print(f"Resized {filename} and saved as {os.path.basename(output_path)}")
            
resize_images("F:\Project\DA-GUI\dataset\hh","F:\Project\DA-GUI\dataset\hh_1")