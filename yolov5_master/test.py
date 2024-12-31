import csv
import os


def image_list():
    class_name = '软件212'
    dropdown_items = []

    matching_names = []
    with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['class'] == class_name:
                matching_names.append(row['name'])

    print(matching_names)

    # 从 person.csv 文件中读取 class 列的唯一值
    with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        classes = set()  # 使用集合来存储唯一的 class 值
        for row in reader:
            classes.add(row['class'])

    # 将每个唯一的 class 值转换为下拉菜单项格式
    for class_name in classes:
        dropdown_items.append({"name": class_name, "url": "#"})

    folder_path = 'D:\Pycharm\pycharmProject\yolov5_web\static\pic'  # 指定图片文件夹路径
    image_files = get_image_files(folder_path,matching_names)
    print(image_files)

def get_image_files(folder_path,names):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图片文件扩展名

    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions and file_name in names:
                image_files.append(file)  # 只返回文件名而不是完整路径



    return image_files


image_list()