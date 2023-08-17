import datetime
from random import randint
from random import sample
import re
import traceback
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import os
import glob
from PIL import Image
from .generate_image_anno import UI_COMMON
from .const import *

from albumentations.augmentations.transforms import *
from albumentations.augmentations.blur.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.resize import *
from albumentations.core.composition import OneOf, Compose, BboxParams


class UI_MERGEIMAGE(UI_COMMON):
    def __init__(self):
        super(UI_MERGEIMAGE, self).__init__()
        # self.setupUi(self)
        pass
        
    def mi_initVar(self):
        # self.cm_initVar()
        
        self.mi_input_img = ""
        self.mi_input_label = ""
        self.mi_input_format = 0
        self.mi_output_img = ""
        self.mi_output_label = ""
        self.mi_output_format = 0
        
        self.mi_total_nums = 2000
        self.mi_img_num =[1,3]
        self.mi_img_th = 0.4
        self.mi_input_bg = os.path.join(os.getcwd(), "data",'background')
        self.mi_output_filename = "{filename}"
        self.mi_img_aug = """[
RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.3, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=False),
Rotate(always_apply=False, p=0.5, limit=(-15, 15), interpolation=1, border_mode=0, value=None, mask_value=None, rotate_method='largest_box', crop_border=False)]"""

    def mi_setVar(self):
        # self.cm_setVar()
        self.sb_mi_img_th.setValue(self.mi_img_th)
        self.sb_mi_img_num0.setValue(self.mi_img_num[0])
        self.sb_mi_img_num1.setValue(self.mi_img_num[1])
        self.sb_mi_total_nums.setValue(self.mi_total_nums)
        
        # 设置默认文件夹
        self.le_mi_input_bg.setText(self.mi_input_bg)
        
        # combobox初始值未变动，手动设置
        self.l_mi_input_format.setText(DATASET_FORMAT[self.mi_input_format])
        self.l_mi_output_format.setText(DATASET_FORMAT[self.mi_output_format])
        self.pte_mi_img_aug.setPlainText(self.mi_img_aug)
        
    def mi_getVar(self):
        # self.cm_getVar()
        self.sb_mi_img_num = [self.sb_mi_img_num0.value(),self.sb_mi_img_num1.value()]
        self.mi_img_th = self.sb_mi_img_th.value()
        self.mi_input_bg = self.le_mi_input_bg.text()
        self.mi_total_nums = self.sb_mi_total_nums.value()
        
        self.mi_input_img = self.l_mi_input_img.text()
        self.mi_input_label = self.l_mi_input_label.text()
        self.mi_input_format = self.l_mi_input_format.text()
        self.mi_output_img = self.l_mi_output_img.text()
        self.mi_output_label = self.l_mi_output_img.text()
        self.mi_output_format = self.l_mi_output_format.text()
        
        self.mi_img_th = 0.4
        self.mi_output_filename = self.l_mi_output_filename.text()
        self.mi_img_aug = self.pte_mi_img_aug.toPlainText()

    def mi_detectVar(self):
        # self.cm_detectVar()
        # 通用配置获取
        self.le_cm_input_img.textChanged.connect(lambda: self.l_mi_input_img.setText(self.le_cm_input_img.text()))
        # 标签
        self.le_cm_input_label.textChanged.connect(lambda: self.l_mi_input_label.setText(self.le_cm_input_label.text()))
        # 格式
        self.cob_cm_input_format.currentIndexChanged.connect(lambda: self.l_mi_input_format.setText(self.cob_cm_input_format.currentText()))
        
        
        self.le_cm_output_img.textChanged.connect(lambda: self.l_mi_output_img.setText(self.le_cm_output_img.text()))
        self.le_cm_output_label.textChanged.connect(lambda: self.l_mi_output_label.setText(self.le_cm_output_label.text()))
        self.cob_cm_output_format.currentIndexChanged.connect(lambda: self.l_mi_output_format.setText(self.cob_cm_output_format.currentText()))
        # 文件名
        self.le_cm_output_filename.textChanged.connect(lambda: self.l_mi_output_filename.setText(self.le_cm_output_filename.text()))
              
        # 按钮
        self.btn_mi_input_bg.clicked.connect(lambda:
            self.le_mi_input_bg.setText(QFileDialog.getExistingDirectory(self, '打开文件夹', self.mi_input_bg)))
        self.btn_mi_do.clicked.connect(self.mi_do)
        
    def mi_do(self):
        self.mi_getVar()

        mi = MERGEIMAGE()
        try:
            mi.merge_images_from_folder(self.mi_input_bg,self.mi_input_img,self.mi_input_label,self.mi_output_img,self.mi_output_label,total_nums=self.mi_total_nums,num=self.mi_img_num,th=0.8,iou_th=self.mi_img_th,max_tries=100,input_format=self.mi_input_format,output_format=self.mi_output_format,out_filename=self.mi_output_filename,aug_list=self.mi_img_aug)
        except Exception as e:
            print(f"合并错误，{e}")
            traceback.print_exc()
        
        print("执行合并")


import random

class MERGEIMAGE(object):
    # 定义所有图片文件的后缀名


    def __init__(self):
        
        pass
      
    @staticmethod
    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算两个框的面积
        area1 = w1 * h1
        area2 = w2 * h2

        # 计算两个框的交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # 计算两个框的并集
        union = area1 + area2 - intersection

        # 计算IOU
        iou = intersection / union

        return iou

    
    def place_fore_into_back(self, back, fore, boxes, th=0.8,iou_th=0.2,max_tries=100):
        # 两个图片之间的iou<0.4,不保证被其他图片遮挡总的的iou 
        back_h, back_w, _ = back.shape
        fore_h, fore_w, _ = fore.shape
        if back_w<fore_w or back_h<fore_h:
            print("background size less than foreground")
            return None, None
        for _ in range(max_tries):
            x = np.random.randint(0, back_w - fore_w)
            y = np.random.randint(0, back_h - fore_h)

            iou_ok = True
            for box in boxes:
                iou_val = self.iou(box, [x, y, x + fore_w, y + fore_h])
                if iou_val >= iou_th:
                    iou_ok = False
                    break

            if iou_ok:
                fore_roi = back[y:y+fore_h, x:x+fore_w, :]
                area_overlap = (fore_roi > 0).all(axis=2).sum()
                area_total = fore_h * fore_w
                if area_overlap / area_total >= th:
                    back[y:y+fore_h, x:x+fore_w, :] = fore
                    return x, y

        return None, None
    
    def merge_images(self,back_img,fore_imgs,nums=2, th=0.8, iou_th=0.2, max_tries = 100):
        try:
           
            # 构造模拟的box集合
            boxes = []

            # 需要插入的前景图数量
            num_fores = nums

            for i in range(num_fores):
                fore_img = fore_imgs[i]
                print(f'Inserting foreground {i+1}/{num_fores}...')
                # print(f"background size:{back_img.size[1:]}, foreground size:{fore_img.size[1:]}")
                # 调用函数将前景图插入到背景图中
                fore_x, fore_y = self.place_fore_into_back(back_img, fore_img, boxes, th=th, iou_th=iou_th, max_tries= max_tries)

                # 如果无法找到合适的位置，则跳过该前景图
                if fore_x is None or fore_y is None:
                    print(f'Could not place foreground {i+1}')
                    # 留位置
                    boxes.append("")
                    continue

                # 将前景图的位置记录到box中
                boxes.append([fore_x, fore_y, fore_x + fore_img.shape[1], fore_y + fore_img.shape[0]])
        except Exception as e:
            print(f"合并多个图片：{e}")
        return back_img,boxes
    
    @staticmethod
    def generate_filename(filename_str,filename="",index=""):
        # 获取当前日期和时间
        now = datetime.datetime.now()
        # 格式化日期和时间字符串
        date = now.strftime("%Y%m%d")
        time = now.strftime("%H%M%S")
        # 构建新的文件名字符串
        new_filename = filename_str.replace("{filename}",filename)
        new_filename = new_filename.replace("{index}",str(index))
        new_filename = new_filename.replace("{date}",date)
        new_filename = new_filename.replace("{time}",time)

        return new_filename
    
    def merge_images_from_folder(self,back_root,fore_root,label_root,ouput_image,output_label,total_nums=10,num=[1,3],th=0.8,iou_th=0.2,max_tries=100,input_format="yolo",output_format="yolo",out_filename = "",aug_list=""):
        back_lists = [i for i in os.listdir(back_root) if i.lower().endswith(tuple(image_extensions))]
        fore_lists = [i for i in os.listdir(fore_root) if i.lower().endswith(tuple(image_extensions))]
        label_lists = [i for i in os.listdir(label_root) if i.lower().endswith(DATASET_FORMAT_EXT[input_format])]
        # 获取列表

        back_indexs = np.random.randint(0, len(back_lists), size=total_nums)
        back_paths = [back_lists[i] for i in back_indexs]
        
        for i_path, back_path in enumerate(back_paths):
            back_img = cv2.imread(os.path.join(back_root,back_path))
            sample_num = sample(range(num[0],num[1]+1),1)[0]
            fore_indexs = np.random.randint(0, len(fore_lists), size=sample_num)
            fore_img_paths = [fore_lists[i] for i in fore_indexs]
            
            # fore_imgs = [ for i in fore_img_paths]
            
            
            # 获取可用数据，包含标签的
            fore_imgs = []
            fore_labels = []
            for i in fore_img_paths:
                path = os.path.splitext(os.path.basename(i))[0]+DATASET_FORMAT_EXT[input_format]
                if path in label_lists:
                    fore_labels.append(os.path.join(label_root,path))
                    # rgba_image = self.rgba_to_rgb_white_background(os.path.join(fore_root,i))
                    rgba_image = cv2.imread(os.path.join(fore_root,i),cv2.IMREAD_UNCHANGED)
                    if rgba_image.shape[2] == 4:
                        rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
                    fore_imgs.append(rgba_image)
                else:
                    print(f"{os.path.join(label_root,path)} not exist")
           
            labels_info  = []
            for i in fore_labels:
                if input_format == "yolo":
                    with open(i,'r') as f:
                        # 标签只有一个
                        label_info = f.read().strip().split(' ')
                        labels_info.append(label_info)
                else:
                    return None

            
            # 增强方式
            if isinstance(aug_list,str) and aug_list != "":
                aug_list = eval(aug_list)
                
            # 这边增加单张图片多目标的裁剪操作
            if isinstance(aug_list,list) and len(aug_list)>0:
                for i in range(len(fore_imgs)):
                    image = fore_imgs[i]
                    anno_label = labels_info[i][0]
                    anno_bbox = [float(i) for i in labels_info[i][1:]]
                    result = Compose(aug_list,bbox_params=BboxParams(format=input_format, min_area=1024, min_visibility=0.1, label_fields=['class_labels']))(image=image,bboxes=[anno_bbox],class_labels=[anno_label])
                    fore_imgs[i] = result["image"]
                    labels_info[i] = [result["class_labels"][0],result["bboxes"][0][0],result["bboxes"][0][1],result["bboxes"][0][2],result["bboxes"][0][3]]
            
            merge_img,box = self.merge_images(back_img,fore_imgs,sample_num,th=th,iou_th = iou_th,max_tries = max_tries)
            bboxs = []
            for i in range(len(box)):
                if box[i] !="":
                    label = labels_info[i][0]
                    x = float(labels_info[i][1])*fore_imgs[i].shape[1]
                    y = float(labels_info[i][2])*fore_imgs[i].shape[0]
                    w = float(labels_info[i][3])*fore_imgs[i].shape[1]
                    h = float(labels_info[i][4])*fore_imgs[i].shape[0]
                    bbox = [(box[i][0]+x)/back_img.shape[1],(box[i][1]+y)/back_img.shape[0], w/back_img.shape[1],h/back_img.shape[0]]
                    bboxs.append([int(label),*bbox])
                else:
                    print(f"第{i}张图为空")
            filename = os.path.splitext(os.path.basename(back_path))[0]
            if out_filename != "":
                filename = self.generate_filename(out_filename,filename = filename,index=i_path)

            img_name = os.path.join(ouput_image,filename)
            label_name = os.path.join(output_label,filename)
            
            with open(label_name+".txt", 'w') as f:
                for bb in bboxs:
                    f.writelines(f"{' '.join([str(i) for i in bb])}\n")
            if output_format == 'yolo':
                cv2.imwrite(img_name+".png",merge_img)

import cv2
if __name__ == "__main__":
    # # 读取背景图和前景图
    # back_img = cv2.imread('image/demo.jpg')
    # fore_img = cv2.imread('image/test.png')
    # fore_img = cv2.resize(fore_img, (800, 400), interpolation = cv2.INTER_AREA)
    mi = MERGEIMAGE()
    # image, bbox = mi.merge_images(back_img,[fore_img]*5,3,th=0.8,iou_th = 0.1)
    # cv2.imwrite('image/hh.png',image)
    
    mi.merge_images_from_folder('background','foreground','foreground','output_merge','output_merge',total_nums=10,num=[1,3],th=0.8,iou_th=0.2,max_tries=100,input_format="yolo",output_format="yolo")
    