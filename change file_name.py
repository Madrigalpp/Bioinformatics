import os
import shutil

def process_h5_files(folder_path):
    # 获取文件夹中所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        # 获取子文件夹中所有.h5文件的列表
        h5_files = [f for f in os.listdir(subfolder) if f.endswith('.h5')]

        for h5_file in h5_files:
            # 提取文件名中符号‘-’后的字符
            new_filename = h5_file.split('-')[-1]

            # 构建新的文件路径
            new_filepath = os.path.join(subfolder, new_filename)

            # 移动文件
            shutil.move(os.path.join(subfolder, h5_file), new_filepath)
            print(f'Moved: {h5_file} to {new_filepath}')

# 指定data文件夹的路径
data_folder_path = 'K_fold5_data'

# 调用函数处理.h5文件
process_h5_files(data_folder_path)
