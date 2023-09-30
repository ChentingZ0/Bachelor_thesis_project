
import os
# 删除无遮挡的原图
file_paths2 = []
for dirpath, dirnames, filenames in os.walk('lfw_masked'):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)  # 构建文件的完整路径
        file_paths2.append(file_path)

files_remove = file_paths2[::2]
print(file_paths2[::2])

for file_remove in files_remove:
    if os.path.exists(file_remove):
        os.remove(file_remove)


# 将遮挡图像改名
files_keep = []
substrings = ["_inpaint", "_surgical", "_KN95", "_empty", "_cloth", "_surgical_blue", "_surgical_green", "_N95", "_gas"]
for dirpath, dirnames, filenames in os.walk('lfw_samples'):
    for old_filename in filenames:
        # rename
        # 提取文件名和扩展名
        file_name, file_extension = os.path.splitext(old_filename)

        for substring in substrings:
            if substring in file_name:
                file_name_new = file_name.replace(substring, "")

        new_filename = file_name_new + file_extension

        old_filepath = os.path.join(dirpath, old_filename)
        new_filepath = os.path.join(dirpath, new_filename)
        # 重命名文件
        os.rename(old_filepath, new_filepath)