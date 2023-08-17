from random import randint
import re
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from .ui_mainwindows import Ui_MainWindow  # 调用生成的.py文件
from .const import DATASET_FORMAT, DATASET_FORMAT_EXT, image_extensions
import os
import glob
from PIL import Image

# class CONST(object):
#     DATASET_FORMAT = ["yolo","coco","pascal_voc"] #{'0':"yolo",'1':"coco",'2':"pascal_voc"}
#     DATASET_FORMAT_EXT = {"yolo":".txt","coco":".json","pascal_voc":".xml"}
#     def __init__(self) -> None:
#         super(CONST, self).__init__()
#         pass

class UTILS(object):
    def __init__(self) -> None:
        super(UTILS, self).__init__()
        pass
    
    @staticmethod
    def get_image_lists(path):
        imglists = []
        # 获取所有图片文件的路径
        image_files = glob.glob(os.path.join(path, "*"))

        # 遍历所有图片文件路径
        for file in image_files:
            # 获取文件扩展名
            ext = os.path.splitext(file)[1]
            # 判断文件扩展名是否为图片格式
            if ext.lower() in image_extensions:
                imglists.append(file)
        
        return imglists
   
class UI_COMMON(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(UI_COMMON, self).__init__()
        pass
    def cm_initVar(self):
        self.cm_input_img = os.path.join(os.getcwd(), "data",'dataset_anno','images') #'image')
        self.cm_input_label = os.path.join(os.getcwd(), "data",'dataset_merge','labels') #,'label')
        self.cm_format = 0
        self.cm_output_img = os.path.join(os.getcwd(), "data",'output_merge','images') #'output_anno_image')
        self.cm_output_label = os.path.join(os.getcwd(), "data",'dataset_anno','labels') #,'output_anno_label')
        
        self.cm_input_format = 0
        self.cm_output_format = 0
        self.cm_output_filename = "aug0_{filename}"
        
        
    def cm_setVar(self):
        self.le_cm_input_img.setText(self.cm_input_img)
        self.le_cm_input_label.setText(self.cm_input_label)
        self.le_cm_output_img.setText(self.cm_output_img)
        self.le_cm_output_label.setText(self.cm_output_label)
        self.cob_cm_input_format.setCurrentIndex(self.cm_input_format)
        self.cob_cm_output_format.setCurrentIndex(self.cm_output_format)
        self.le_cm_output_filename.setText(self.cm_output_filename)
        
    def cm_getVar(self):
        self.cm_input_img = self.le_cm_input_img.text()
        self.cm_input_label = self.le_cm_input_label.text()
        self.cm_output_img = self.le_cm_output_img.text()
        self.cm_output_label = self.le_cm_output_label.text()
        self.cm_input_format = self.cob_cm_input_format.currentIndex()
        self.cm_output_format = self.cob_cm_output_format.currentIndex()
        self.cm_output_filename = self.le_cm_output_filename.text()
    
    def cm_detectVar(self):
        # 输入文件按钮
        self.btn_cm_input_img.clicked.connect(self.open_cm_input_img)
        self.btn_cm_input_label.clicked.connect(self.open_cm_input_label)
            #lambda: self.le_cm_input_label.setText(QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_input_label)))
        # 输出文件按钮
        self.btn_cm_output_img.clicked.connect(self.open_cm_output_img)
            # lambda: self.le_cm_output_img.setText(QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_output_img))) 
        # 输出文件按钮
        self.btn_cm_output_label.clicked.connect(self.open_cm_output_label)
            # lambda: self.le_cm_output_label.setText(QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_output_label))) 

    def open_cm_input_img(self):
        img_dir =  QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_input_img)
        if img_dir!="":
            self.cm_input_img = img_dir
            self.le_cm_input_img.setText(self.cm_input_img)
    def open_cm_output_img(self):
        img_dir =  QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_output_img)
        if img_dir!="":
            self.cm_output_img = img_dir
            self.le_cm_output_img.setText(self.cm_output_img)
    def open_cm_input_label(self):
        img_dir =  QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_input_label)
        if img_dir!="":
            self.cm_input_label = img_dir
        self.le_cm_input_label.setText(self.cm_input_label)
    def open_cm_output_label(self):
        img_dir =  QFileDialog.getExistingDirectory(self, '打开文件夹', self.cm_output_label)
        if img_dir!="":
            self.cm_output_label = img_dir
        self.le_cm_output_label.setText(self.cm_output_label)

# 单张图片标注
class UI_IMAGEANNO(UTILS,UI_COMMON):
    def __init__(self):
        super(UI_IMAGEANNO, self).__init__()
        # self.setupUi(self)
        pass
        
    def ia_initVar(self):
        # self.cm_initVar()
        
        self.ia_kind = "营业执照"
        self.ia_kinds = "圆形印章 椭圆印章 身份证 营业执照 银行卡"
        self.ia_format = 0

    def ia_setVar(self):
        # self.cm_setVar()
        self.le_anno_kinds.setText(self.ia_kinds)
        self.le_anno_kind.setText(self.ia_kind)
        self.l_anno_format.setText(DATASET_FORMAT[self.ia_format])
        
    def ia_getVar(self):
        # self.cm_getVar()
        self.ia_input = self.l_anno_input.text()
        self.ia_kinds = self.le_anno_kinds.text()
        self.ia_kind = self.le_anno_kind.text()
        self.ia_output = self.l_anno_output.text()
        self.ia_format = self.l_anno_format.text()

    def ia_detectVar(self):
        # self.cm_detectVar()
        # 通用配置获取
        self.le_cm_input_img.textChanged.connect(lambda: self.l_anno_input.setText(self.le_cm_input_img.text()))
        self.le_cm_output_label.textChanged.connect(lambda: self.l_anno_output.setText(self.le_cm_output_label.text()))
        self.cob_cm_output_format.currentIndexChanged.connect(lambda: self.l_anno_format.setText(self.cob_cm_output_format.currentText()))
        
        # 按钮
        self.btn_anno_trans.clicked.connect(self.ia_anno_trans)
        
    def ia_anno_trans(self):
        self.ia_getVar()
        data_format = self.ia_format
        imglists = self.get_image_lists(self.ia_input)
        labels = [i for i in re.split(r"[\s,;]+",self.ia_kinds) if i.strip()!=""]
        label_index = labels.index(self.ia_kind)
        for img in imglists:
            filename, ext = os.path.splitext(os.path.basename(img))
            wh = Image.open(img).size
            bbox = self.generate_image_box(wh,format = data_format)
            label_info = self.generate_image_label(bbox,label_index,format=data_format)
            if not os.path.exists(self.ia_output):
                os.makedirs(self.ia_output)
            with open(os.path.join(self.ia_output,filename+DATASET_FORMAT_EXT[data_format]),"w") as f:
                f.write(label_info)

        print("转换完成了")
        

    def generate_image_label(self,bbox,label_id,format="yolo"):
        if format == "yolo":
            return f"{label_id} {' '.join([str(i) for i in bbox])}"

    def generate_image_box(self,wh,format="yolo"):
        xc = randint(490, 500)/1000
        yc = randint(490, 500)/1000
        w = randint(xc*2000-20, xc*2000)/1000
        h = randint(yc*2000-20, yc*2000)/1000
        xmin = xc - w/2
        ymin = yc - h/2
        if format == "yolo":
            return (xc,yc,w,h)
        if format == "coco":
            return [int(x) for x in (wh[0]*xmin,wh[1]*ymin,wh[0]*w,wh[1]*h)]
        if format == "pascal_voc":
            return [int(x) for x in (wh[0]*xmin,wh[1]*ymin,wh[0]*(xmin+w),wh[1]*(ymin+h))]
    
    
        

    