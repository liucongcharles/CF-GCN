import os
import shutil

def find_files_from_txt(txt_file, folder_path, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 读取txt文件内容
    with open(txt_file, 'r') as file:
        txt_content = file.readlines()

    # 遍历文件夹中的文件，找出所有匹配的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.strip() in [line.strip() for line in txt_content]:
                # 复制文件到输出文件夹
                shutil.copy(os.path.join(root, file), output_folder)

# 示例用法
txt_file = '/Users/liucong/Downloads/DSIFN-CD-256/list/test.txt'
folder_path = '/Users/liucong/Downloads/DSIFN-CD-256/label'
output_folder = '/Users/liucong/Downloads/DSIFN-CD-Star/test/label'
find_files_from_txt(txt_file, folder_path, output_folder)


