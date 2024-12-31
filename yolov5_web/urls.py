"""yolov5_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from django.contrib import staticfiles
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # 这里的 'home' 是视图函数的名称
    path('home/', views.home, name='home'),
    path('detect/', views.detect, name='detect'),
    path('detectp/', views.detectp, name='detectp'),
    path('test/', views.test, name='test'),
    path('detectd/', views.detectd, name='detectd'),
    path('face/', views.face, name='face'),
    path('facep/', views.facep, name='facep'),
    path('addperson/', views.addperson, name='addperson'),
    path('changeperson/', views.changeperson, name='changeperson'),
    path('findperson/', views.findperson, name='findperson'),
    path('deleteperson/', views.deleteperson, name='deleteperson'),
    path('showResult/', views.showResult, name='showResult'),
    path('image_list/', views.image_list, name='image_list'),
    path('person_result/', views.person_result, name='person_result'),
    path('result_list/', views.result_list, name='result_list'),
    path('result_result/', views.result_result, name='result_result'),
    path('video_list/', views.video_list, name='video_list'),
    path('history/', views.history, name='history'),
    path('chart_view/', views.chart_view, name='chart_view'),
    path('save_device_ip/', views.save_device_ip, name='save_device_ip'),
    path('output_image/', views.output_image, name='output_image'),
    path('convert_csv_to_excel/', views.convert_csv_to_excel, name='convert_csv_to_excel'),
    path('grade_list/', views.grade_list, name='grade_list'),
    path('grade_result/', views.grade_result, name='grade_result'),
    path('update_data/', views.update_data, name='update_data'),
]
urlpatterns += staticfiles_urlpatterns()

