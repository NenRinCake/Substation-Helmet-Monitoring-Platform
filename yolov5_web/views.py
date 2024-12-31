import glob
import json
import os
import shutil

import pandas as pd
import pypinyin
import requests
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import subprocess
def home(request):

    return render(request, 'index.html')
# views.py

def detect(request):
    subprocess.run(['python', '.\yolov5_master\detect.py'], check=True)
    return render(request, 'index.html')

def detectp(request):
    subprocess.run(['python', '.\yolov5_master\detectp.py'], check=True)
    return render(request, 'show.html')

def detectd(request):
    subprocess.run(['python', '.\yolov5_master\detectd.py'], check=False)
    return render(request, 'index.html')

def face(request):
    subprocess.run(['python', '.\yolov5_master\\face.py'], check=False)
    return render(request, 'index.html')

def addperson(request):
    subprocess.run(['python', '.\yolov5_master\\addperson.py'], check=False)
    return render(request, 'index.html')

def changeperson(request):
    if request.method == 'POST':
        ID = request.POST.get('ID', '')
        name = request.POST.get('Sname', '')
        gender = request.POST.get('gender', '')
        Class = request.POST.get('class', '')
        photo = request.FILES.get('studentPhoto')
        print(photo)
        pinyin_name = chinese_name_to_english(name)
        # 假设 CSV 文件路径为 '/path/to/your/csv/file.csv'
        fieldnames = ['ID', 'name', 'class', 'gender', 'score', 'text', 'CNname']
        file_path = 'D:\Pycharm\pycharmProject\yolov5_web\person\person.csv'
        with open(file_path, 'a', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'ID': ID, 'name': pinyin_name, 'class': Class, 'gender': gender, 'score': 100, 'text': '', 'CNname': name})

    class_name = request.GET.get('name', '')
    # with open('D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\default_class.txt', 'r', encoding='utf-8') as file:
    #     default_class = file.read()
    # class_name = default_class
    dropdown_items = []
    if class_name == '':
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
        image_files = get_file_name(folder_path)

    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    source_files = [f for f in os.listdir(desktop_path) if f.startswith(pinyin_name) and f.lower().endswith('.jpg')]

    for file in source_files:
        source_file_path = os.path.join(desktop_path, file)
        destination_file_path = os.path.join(folder_path, file)
        shutil.copy(source_file_path, destination_file_path)


    return render(request, 'personList.html', {'image_files': image_files, 'dropdown_items': dropdown_items})


def chinese_name_to_english(chinese_name):
    # 将中文姓名分割成姓和名
    last_name = chinese_name[0]
    first_name = chinese_name[1:]

    # 将姓转换为拼音并取第一个字母大写
    english_last_name = pypinyin.lazy_pinyin(last_name)[0].capitalize()

    # 将名转换为拼音并取每个拼音的首字母大写
    english_first_name = ''.join([pinyin.capitalize() for pinyin in pypinyin.lazy_pinyin(first_name)]).title()

    # 输出英文姓名
    english_name = english_last_name + ' ' + english_first_name
    return english_name

def findperson(request):
    if request.method == 'POST':
        image_files = []
        ID = request.POST.get('ID', '')
        # 假设 CSV 文件路径为 '/path/to/your/csv/file.csv'
        file_path = 'D:\\Pycharm\\pycharmProject\\yolov5_web\\person\\person.csv'
        information = {}

        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] == ID:
                    # 匹配成功则将该行信息添加到字典中
                    information['ID'] = row['ID']
                    information['name'] = row['name']
                    information['gender'] = row['gender']
                    information['class'] = row['class']
                    information['CNname'] = row['CNname']

                    # 查找对应的照片文件
                    folder_path = 'D:\\Pycharm\\pycharmProject\\yolov5_web\\static\\pic'
                    for filename in os.listdir(folder_path):
                        if filename.startswith(row['name']):
                            image_files.append(filename)

                    break  # 匹配成功后直接退出循环

    if not information:
        return render(request, 'index.html', {'information_not_found': True})

    return render(request, 'find_person.html', {'image_files': image_files, 'information': information})


def deleteperson(request):
    if request.method == 'POST':
        student_id = request.POST.get('studentID', None)

        # 假设 CSV 文件路径为 '/path/to/your/csv/file.csv'
        file_path = 'D:\Pycharm\pycharmProject\yolov5_web\person\person.csv'

        new_data = []  # 用于存储更新后的所有行数据
        name = []
        # 读取原始CSV文件内容，并进行更新
        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames  # 获取CSV文件的字段名
            for row in reader:
                if row['ID'] == student_id:
                    name = row['name']
                    # 更新匹配ID的行的score和text字段值
                    continue
                new_data.append(row)
        # 将更新后的数据写回CSV文件
        with open(file_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # 写入CSV文件的表头
            writer.writerows(new_data)  # 写入更新后的所有行数据




    class_name = request.GET.get('name', '')
    # with open('D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\default_class.txt', 'r', encoding='utf-8') as file:
    #     default_class = file.read()
    # class_name = default_class
    dropdown_items = []
    if class_name == '':
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
        image_files = get_file_name(folder_path)

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 检查文件是否以指定名称开头且以 '.jpg' 结尾
                if file.startswith(name) and file.lower().endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    # 删除文件
                    os.remove(file_path)

        return render(request, 'personList.html', {'image_files': image_files, 'dropdown_items': dropdown_items})


def showResult(request):
    file_path = 'D:\Pycharm\pycharmProject\yolov5_web\device\device.csv'

    dropdown_items = []
    ip_items = []
    # 从 person.csv 文件中读取 class 列的唯一值
    with open('D:\Pycharm\pycharmProject\yolov5_web\device\device.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        classes = set()  # 使用集合来存储唯一的 class 值
        for row in reader:
            classes.add(row['place'])

    # 将每个唯一的 class 值转换为下拉菜单项格式
    for class_name in classes:
        dropdown_items.append({"place": class_name, "url": "#"})
    subprocess.run(['python', '.\yolov5_master\\showResult.py'], check=False)
    folder_path = 'D:\Pycharm\pycharmProject\yolov5_web\static\\result'  # 指定图片文件夹路径
    image_files = get_file_count(folder_path)
    return render(request, 'resultList.html',{'image_files': image_files,'dropdown_items': dropdown_items})


def test(request):
    return render(request, 'test.html')

def facep(request):
    subprocess.run(['python', '.\yolov5_master\\facep.py'], check=False)
    return render(request, 'show.html')

def image_list(request):
    class_name = request.GET.get('name', '')
    # with open('D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\default_class.txt', 'r', encoding='utf-8') as file:
    #     default_class = file.read()
    # class_name = default_class
    dropdown_items = []
    image_files = []
    if class_name == '':
        # 从 person.csv 文件中读取 class 列的唯一值
        with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            classes = set()  # 使用集合来存储唯一的 class 值
            for row in reader:
                classes.add(row['class'])
                image_files.append(row['CNname'])

        # 将每个唯一的 class 值转换为下拉菜单项格式
        for class_name in classes:
            dropdown_items.append({"name": class_name, "url": "#"})

        folder_path = 'D:\Pycharm\pycharmProject\yolov5_web\static\pic'  # 指定图片文件夹路径
        image_files = get_file_name(folder_path)
        return render(request, 'personList.html', {'image_files': image_files, 'dropdown_items': dropdown_items})

def get_result_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图片文件扩展名

    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(file)  # 只返回文件名而不是完整路径

    return image_files


def get_file_count(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图片文件扩展名

    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                head_num = "0"
                if "h" in file:
                    head_num = file[-5]
                image_files.append({'file': file, 'count': head_num + "人未佩戴安全头盔"})  # 只返回文件名而不是完整路径

    return image_files

def person_result(request):
    if request.method == 'POST':
        # 获取通过表单提交的数据
        class_name = request.POST.get('class_name', '')
    dropdown_items = []
    matching_names = []
    person_ID = []
    with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['class'] == class_name:
                matching_names.append(row['name'])
                person_ID.append(row['ID'])

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
    image_files = get_image_files(folder_path, matching_names)

    data = zip(image_files, person_ID)
    print(data)
    return render(request, 'person_result.html', {'data': data, 'dropdown_items': dropdown_items,})

def get_image_files(folder_path,names):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图片文件扩展名

    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions and file_name in names:
                # image_files.append(file)  # 只返回文件名而不是完整路径
                name_without_extension = os.path.splitext(file)[0]
                with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as files1:
                    reader = csv.DictReader(files1)
                    for row in reader:
                        if row['name'] == name_without_extension:
                            image_files.append({'file': file, 'name': row['CNname']})

    return image_files

def result_list(request):
    file_path = 'D:\Pycharm\pycharmProject\yolov5_web\device\device.csv'

    dropdown_items = []
    ip_items = []
    # 从 person.csv 文件中读取 class 列的唯一值
    with open('D:\Pycharm\pycharmProject\yolov5_web\device\device.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        classes = set()  # 使用集合来存储唯一的 class 值
        for row in reader:
            classes.add(row['place'])

    # 将每个唯一的 class 值转换为下拉菜单项格式
    for class_name in classes:
        dropdown_items.append({"place": class_name, "url": "#"})


    folder_path = 'D:\Pycharm\pycharmProject\yolov5_web\static\\result'  # 指定图片文件夹路径
    image_files = get_file_count(folder_path)
    print(image_files)

    return render(request, 'resultList.html', {'image_files': image_files, 'dropdown_items': dropdown_items})

def result_result(request):
    if request.method == 'POST':
        # 获取通过表单提交的数据
        class_name = request.POST.get('class_name', '')
    dropdown_items = []
    new_data = []
    with open('D:\Pycharm\pycharmProject\yolov5_web\device\device.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames  # 获取CSV文件的字段名
        for row in reader:
            if row['place'] == class_name:
                # 更新匹配ID的行的score和text字段值
                new_data.append(row['IP'])

    piece = ""
    for IP in new_data:
        for i in IP:
            if i == ":":
                break
            else:
                piece += i
    # 从 person.csv 文件中读取 class 列的唯一值
    with open('D:\Pycharm\pycharmProject\yolov5_web\device\device.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        classes = set()  # 使用集合来存储唯一的 class 值
        for row in reader:
            classes.add(row['place'])
    # 将每个唯一的 class 值转换为下拉菜单项格式
    for class_name in classes:
        dropdown_items.append({"place": class_name, "url": "#"})
    print(piece)
    folder_path = 'D:\Pycharm\pycharmProject\yolov5_web\static\\result'  # 指定图片文件夹路径
    all_files = os.listdir(folder_path)
    image_files = []
    # 筛选文件名中包含在new_data列表中的文件
    for i in all_files:
        if piece in i:
            head_num = "0"
            if "h" in i:
                head_num = i[-5]
            image_files.append({'file': i, 'count': head_num + "人未佩戴安全头盔"})
    print(image_files)

    return render(request, 'result_result.html', {'dropdown_items': dropdown_items,'image_files': image_files})

def get_file_name(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图片文件扩展名

    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                # image_files.append({})  # 只返回文件名而不是完整路径
                name_without_extension = os.path.splitext(file)[0]
                with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as files1:
                    reader = csv.DictReader(files1)
                    for row in reader:
                        if row['name'] == name_without_extension:
                            image_files.append({'file':file,'name':row['CNname']})

    # print(image_files)
    return image_files


# views.py

from django.shortcuts import render
import csv

def history(request):
    history_file_path = 'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\historyDevice.csv'
    history_data = []
    file_path = 'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\DeviceIP.txt'
    with open(file_path, 'r') as file:
        id = file.read()
    with open(history_file_path, 'r') as history_file:
        csv_reader = csv.DictReader(history_file)
        for row in csv_reader:
            device_ip = row['工作设备IP']
            start_time = row['开始工作时间']
            end_time = row['结束工作时间']
            work_time = row['设备工作时长(分钟)']
            place = row['地点']
            history_data.append(
                {'device_ip': device_ip, 'start_time': start_time, 'end_time': end_time, 'work_time': work_time, 'place': place})

    return render(request, 'history.html', {'history_data': history_data, 'id':id})

def save_device_ip(request):
    if request.method == 'POST':
        device_ip = request.POST.get('device_ip', '')  # 获取设备IP
        if device_ip:  # 如果设备IP非空
            # 保存设备IP到txt文件
            file_path = 'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\DeviceIP.txt'
            with open(file_path, 'w') as file:
                file.write(device_ip)

                # 检查并插入设备IP到CSV文件
            history_file_path = 'D:\Pycharm\pycharmProject\yolov5_web\device\device.csv'
            with open(history_file_path, 'r', encoding='utf-8', newline='') as history_file:
                csv_reader = csv.DictReader(history_file)
                ip_exists = False
                for row in csv_reader:
                    if row['IP'] == device_ip:
                        ip_exists = True
                        break

                if not ip_exists:
                    # 插入设备IP到CSV文件
                    with open(history_file_path, 'a', newline='') as history_file:
                        fieldnames = ['IP']
                        csv_writer = csv.DictWriter(history_file, fieldnames=fieldnames)
                        csv_writer.writerow({'IP': device_ip})

    history_file_path = 'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\historyDevice.csv'
    history_data = []
    file_path = 'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\DeviceIP.txt'
    with open(file_path, 'r') as file:
        id = file.read()
    with open(history_file_path, 'r') as history_file:
        csv_reader = csv.DictReader(history_file)
        for row in csv_reader:
            device_ip = row['工作设备IP']
            start_time = row['开始工作时间']
            end_time = row['结束工作时间']
            work_time = row['设备工作时长(分钟)']
            place = row['地点']
            history_data.append(
                {'device_ip': device_ip, 'start_time': start_time, 'end_time': end_time, 'work_time': work_time, 'place': place})

    # 处理完逻辑后重定向回原页面
    return render(request,'history.html', {'history_data': history_data, 'id':id} )


def read_csv(file_path):
    head_count = 0
    helmet_count = 0

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for item in row:
                if item.strip().lower() == 'head':
                    head_count += 1
                elif item.strip().lower() == 'helmet':
                    helmet_count += 1

    return head_count, helmet_count


def process_last_5_csv_files(folder_path):
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')), key=os.path.getmtime, reverse=True)
    last_5_files = csv_files[:5]

    results = []

    for file_path in last_5_files:
        head_count, helmet_count = read_csv(file_path)
        file_name = os.path.basename(file_path)
        percentage = round((helmet_count / (head_count + helmet_count)) * 100, 2)
        results.append({'File': file_name, 'Percentage': percentage})

    return results

def process_last_5_csv_files_name_count(folder_path):
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')), key=os.path.getmtime, reverse=True)
    last_5_files = csv_files[:6]

    results = []

    for file_path in last_5_files:
        unique_names = set()  # Use a set to store unique names
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = row.get('Name')  # Assuming the column header is 'Name'
                if name:
                    unique_names.add(name.strip())  # Add the name to the set

        file_name = os.path.basename(file_path)
        name_count = len(unique_names)
        results.append({'File': file_name, 'Count': name_count})

    return results

def process_device_usage_csv(file_path):
    device_usage = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            location = row.get('地点')
            duration = float(row.get('设备工作时长(分钟)', 0))
            if location:
                device_usage[location] = device_usage.get(location, 0) + duration
    return [{'location': location, 'duration': duration} for location, duration in device_usage.items()]


def chart_view(request):
    folder_path_percentage = r'D:\Pycharm\pycharmProject\yolov5_web\safe or dangerous'
    folder_path_name_count = r'D:\Pycharm\pycharmProject\yolov5_web\information'
    file_results_percentage = process_last_5_csv_files(folder_path_percentage)
    file_results_name_count = process_last_5_csv_files_name_count(folder_path_name_count)
    file_results_device_use = r'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\historyDevice.csv'

    for result in file_results_percentage:
        result['File'] = os.path.splitext(result['File'])[0]

    for result in file_results_name_count:
        result['File'] = os.path.splitext(result['File'])[0]

    chart_data_percentage = json.dumps(file_results_percentage)
    chart_data_name_count = json.dumps(file_results_name_count)

    device_usage_data = process_device_usage_csv(file_results_device_use)
    device_usage_data_json = json.dumps(device_usage_data)
    # print(chart_data_percentage)
    # print(chart_data_name_count)
    # print(device_usage_data)


    return render(request, 'chart.html',
                  {'chart_data_percentage': chart_data_percentage, 'chart_data_name_count': chart_data_name_count,
                   'device_usage_data_json': device_usage_data_json})



def output_image(request):
    if request.method == 'POST':
        try:
            # 从请求体中获取JSON数据
            data = json.loads(request.body)
            chart_data = data.get('chartData', None)
            chart_data3 = data.get('chartData3', None)
            chart_data1 = data.get('chartData1', None)

            # 创建保存文件夹（如果不存在）
            save_folder = 'D:\Pycharm\pycharmProject\yolov5_web\excel'
            os.makedirs(save_folder, exist_ok=True)

            # 将数据转换为DataFrame
            df_chart_data = pd.DataFrame(chart_data)
            df_chart_data3 = pd.DataFrame(chart_data3)
            df_chart_data1 = pd.DataFrame(chart_data1)

            # 将DataFrame写入Excel文件
            df_chart_data.to_excel(os.path.join(save_folder, 'Helmet_wear_rate.xlsx'), index=False)
            df_chart_data3.to_excel(os.path.join(save_folder, 'Duration_of_device_use.xlsx'), index=False)
            df_chart_data1.to_excel(os.path.join(save_folder, 'Number_of_attendees.xlsx'), index=False)

            print("Excel文件已保存到文件夹:", save_folder)


        except json.JSONDecodeError:
            print("Error decoding JSON data")

    return render(request, 'index.html')



def convert_csv_to_excel(request):
    folder_path = r'D:\Pycharm\pycharmProject\yolov5_web\information'  # 指定文件夹路径

    # 获取文件夹中的所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # 遍历每个CSV文件，将其转换为Excel格式并保存
    for csv_file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 获取CSV文件名
        file_name = os.path.splitext(os.path.basename(csv_file))[0]

        # 创建Excel文件名
        output_folder = r'D:\Pycharm\pycharmProject\yolov5_web\excel'
        # 将数据写入Excel文件
        # 创建Excel文件名
        excel_file = os.path.join(output_folder, f'{file_name}.xlsx')

        # 将数据写入Excel文件
        df.to_excel(excel_file, index=False)

    folder_path_percentage = r'D:\Pycharm\pycharmProject\yolov5_web\safe or dangerous'
    folder_path_name_count = r'D:\Pycharm\pycharmProject\yolov5_web\information'
    file_results_percentage = process_last_5_csv_files(folder_path_percentage)
    file_results_name_count = process_last_5_csv_files_name_count(folder_path_name_count)
    file_results_device_use = r'D:\Pycharm\pycharmProject\yolov5_web\yolov5_master\historyDevice.csv'

    for result in file_results_percentage:
        result['File'] = os.path.splitext(result['File'])[0]

    for result in file_results_name_count:
        result['File'] = os.path.splitext(result['File'])[0]

    chart_data_percentage = json.dumps(file_results_percentage)
    chart_data_name_count = json.dumps(file_results_name_count)

    device_usage_data = process_device_usage_csv(file_results_device_use)
    device_usage_data_json = json.dumps(device_usage_data)

    # 返回成功消息给前端
    return render(request,'chart.html',{'chart_data_percentage': chart_data_percentage, 'chart_data_name_count': chart_data_name_count,
                                        'device_usage_data_json': device_usage_data_json})


def grade_list(request):
    file_path = 'D:\Pycharm\pycharmProject\yolov5_web\person\person.csv'
    grade_data = []
    with open(file_path, 'r', encoding='utf-8') as grade_file:
        csv_reader = csv.DictReader(grade_file)
        for row in csv_reader:
            ID = row['ID']
            name = row['name']
            class_name = row['class']
            gender = row['gender']
            score = row['score']
            text = row['text']
            CNname = row['CNname']
            grade_data.append({'ID': ID, 'name': name, 'class_name': class_name, 'gender': gender, 'score': score, 'text': text, 'CNname':CNname})

    dropdown_items = []
    matching_names = []
    person_ID = []
    with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['class'] == class_name:
                matching_names.append(row['name'])
                person_ID.append(row['ID'])

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
    image_files = get_image_files(folder_path, matching_names)

    data = zip(image_files, person_ID)

    return render(request, 'gradeList.html', {'grade_data': grade_data, 'dropdown_items':dropdown_items})

def update_data(request):
    if request.method == 'POST':
        ID = request.POST.get('ID', '')
        score = request.POST.get('score', '')
        text = request.POST.get('text', '')
        # 假设 CSV 文件路径为 '/path/to/your/csv/file.csv'
        file_path = 'D:\Pycharm\pycharmProject\yolov5_web\person\person.csv'


        new_data = []  # 用于存储更新后的所有行数据

        # 读取原始CSV文件内容，并进行更新
        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames  # 获取CSV文件的字段名
            for row in reader:
                if row['ID'] == ID:
                    # 更新匹配ID的行的score和text字段值
                    row['ID'] = ID
                    row['score'] = score
                    row['text'] = text
                new_data.append(row)
        print(new_data)
        # 将更新后的数据写回CSV文件
        with open(file_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # 写入CSV文件的表头
            writer.writerows(new_data)  # 写入更新后的所有行数据


    file_path = 'D:\Pycharm\pycharmProject\yolov5_web\person\person.csv'
    grade_data = []
    with open(file_path, 'r', encoding='utf-8') as grade_file:
        csv_reader = csv.DictReader(grade_file)
        for row in csv_reader:
            ID = row['ID']
            name = row['name']
            class_name = row['class']
            gender = row['gender']
            score = row['score']
            text = row['text']
            CNname = row['CNname']
            grade_data.append(
                {'ID': ID, 'name': name, 'class_name': class_name, 'gender': gender, 'score': score, 'text': text, 'CNname':CNname})

        dropdown_items = []
        matching_names = []
        person_ID = []
        with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['class'] == class_name:
                    matching_names.append(row['name'])
                    person_ID.append(row['ID'])

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
        image_files = get_image_files(folder_path, matching_names)

        data = zip(image_files, person_ID)

    return render(request, 'gradeList.html', {'grade_data': grade_data, 'dropdown_items':dropdown_items})


def grade_result(request):
    if request.method == 'POST':
        # 获取通过表单提交的数据
        class_name = request.POST.get('class_name', '')
    dropdown_items = []
    new_data = []
    with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames  # 获取CSV文件的字段名
        for row in reader:
            if row['class'] == class_name:
                # 更新匹配ID的行的score和text字段值
                new_data.append(row)
        print(new_data)
    # 从 person.csv 文件中读取 class 列的唯一值
    with open('D:\Pycharm\pycharmProject\yolov5_web\person\person.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        classes = set()  # 使用集合来存储唯一的 class 值
        for row in reader:
            classes.add(row['class'])

    # 将每个唯一的 class 值转换为下拉菜单项格式
    for class_name in classes:
        dropdown_items.append({"name": class_name, "url": "#"})


    return render(request, 'grade_result.html', {'dropdown_items': dropdown_items,'new_data': new_data})


def video_list(request):
    return render(request, 'videoList.html')

