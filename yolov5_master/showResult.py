import os

def open_folder(path):
    try:
        os.startfile(path)
        print(f"打开文件夹成功: {path}")
    except Exception as e:
        print(f"打开文件夹失败: {e}")

# 要打开的文件夹路径
folder_path = "D:\Pycharm\pycharmProject\yolov5_web\static\\result"

open_folder(folder_path)
