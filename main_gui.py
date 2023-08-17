import datetime
import os
import re
import json
import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
from PyQt5 import QtWidgets

from detaug.ui_mainwindows import Ui_MainWindow  # 调用生成的.py文件
from detaug.dataprocess import EasyDL
image_format = ""

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.blur.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.resize import *
from albumentations.core.composition import OneOf, Compose, BboxParams

import numpy as np
from PIL import Image

from detaug.labeltrans import LabelTran
from detaug.generate_image_anno import UI_IMAGEANNO
from detaug.merge_images import UI_MERGEIMAGE

from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap, QWheelEvent, QTransform
from PyQt5.QtCore import Qt, QObject
import sys
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QEvent
class WheelEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta > 0:
                scaleFactor = 1.1
            else:
                scaleFactor = 1 / 1.1
            obj.setTransform(
                QTransform().scale(scaleFactor, scaleFactor), True
            )
            return True
        return False



class DataAug(UI_IMAGEANNO,UI_MERGEIMAGE):
    interpolation = {'cv2.INTER_NEAREST':0,'cv2.INTER_LINEAR':1,'cv2.INTER_CUBIC':2,'cv2.INTER_AREA':3,'cv2.INTER_LANCZOS4':4}
    border = {'cv2.BORDER_CONSTANT':0,'cv2.BORDER_REFLECT':1,'cv2.BORDER_REFLECT_101':2,'cv2.BORDER_REPLICATE':3,'cv2.BORDER_WRAP':4}
    MODE = {'Compose':0,"Oneof":1}
    MODE1 = {'fast':0,'exact':1}
    DATA_FORMAT = {"yolo":".txt","pascal_voc":".xml","coco":".json","albumentations":".txt"}
    DATASE_FORMAT = {"yolo":0,"coco":1,"pascal_voc":2, "albumentations":3}
    class_labels =['horse1','horse2']
    
    def __init__(self):
        super(DataAug, self).__init__()
        self.v_image_path=""
        self.initVar()
        
        self.setupUi(self)
        self.initUI()
        
        self.setVar()
        
    def export_config(self):
        self.getVar()
        # 按照group编写
        params ={
            "aug":{
                "single_config":{
                    "image_path":self.image_path,
                    "anno_path":self.anno_path,
                    "dataset_format":self.dataset_format,
                    "index":self.aug_index,
                },
                "batch_config":{
                    "image_dir":self.image_folder_path,
                    "anno_dir":self.anno_folder_path,
                    "datasets_format":self.datasets_format,
                    "index":self.aug_index,
                },
                "aug_method":{
                    "advancedblur":{
                        "select": self.advancedblur_select,
                        "mode": self.advancedblur_mode,
                        "blur": self.advancedblur_blur,
                        "sigmax": self.ab_sigmax,
                        "sigmay": self.ab_sigmay,
                        "beta_limit": self.ab_beta_limit,
                        "noise_limit": self.ab_noise_limit,
                        "rotate_limit": self.ab_rotate_limit,
                        "p": self.advancedblur_p,
                    },
                    "blur":{
                        "select": self.blur_select,
                        "mode": self.blur_mode,
                        "blur_limit": self.blur_blur_limit,
                        "p": self.blur_p,
                    },
                    "defocus":{
                        "select": self.defocus_select,
                        "mode": self.defocus_mode,
                        "radius": self.defocus_radius,
                        "alias_blur": self.defocus_alias_blur,
                        "p": self.defocus_p,
                    },
                    "guassianblur":{
                        "select": self.gab_select,
                        "mode": self.gab_mode,
                        "blur_limit": self.gab_blur_limit,
                        "sigma_limit": self.gab_sigma_limit,
                        "p": self.gab_p,
                    },
                    "GlassBlur":{
                        "select": self.glb_select,
                        "mode": self.glb_mode,
                        "sigma": self.glb_sigma,
                        "max_delta": self.glb_max_delta,
                        "iterations": self.glb_iterations,
                        "mode1": self.glb_mode1,
                        "p": self.glb_p,
                    },
                    "MedianBlur":{
                        "select": self.meb_select,
                        "mode": self.meb_mode,
                        "blur_limit": self.meb_blur_limit,
                        "p": self.meb_p,
                    },
                    "MotionBlur":{
                        "select": self.mob_select,
                        "mode": self.mob_mode,
                        "blur_limit": self.mob_blur_limit,
                        "allow_shifted": self.mob_allow_shifted,
                        "p": self.mob_p,
                    },
                    "ZoomBlur":{
                        "select": self.zb_select,
                        "mode": self.zb_mode,
                        "max_factor": self.zb_max_factor,
                        "step_factor": self.zb_step_factor,
                        "p": self.zb_p,
                    },
                    "RandomBrightness":{
                        "select": self.rb_select,
                        "mode": self.rb_mode,
                        "limit": self.rb_limit,
                        "p": self.rb_p,
                    },
                    "RandomBrightnessContrast":{
                        "select": self.rbc_select,
                        "mode": self.rbc_mode,
                        "bl": self.rbc_bl,
                        "cl": self.rbc_cl,
                        "bbm": self.rbc_bbm,
                        "p": self.rbc_p,
                    },
                    # "RandomContrast":{
                    #     "select": self.rc_select,
                    #     "mode": self.rc_mode,
                    #     "limit": self.rc_limit,
                    #     "p": self.rc_p,
                    # },
                    "RandomFog":{
                        "select": self.rf_select,
                        "mode": self.rf_mode,
                        "fcl": self.rf_fcl,
                        "fcu": self.rf_fcu,
                        "ac": self.rf_ac,
                        "p": self.rf_p,
                    },
                    "RandomGamma":{
                        "select": self.rga_select,
                        "mode": self.rga_mode,
                        "gl": self.rga_gl,
                        "p": self.rga_p,
                    },
                    # "RandomGravel":{
                    #     "select": self.rgr_select,
                    #     "mode": self.rgr_mode,
                    #     "gr": self.rgr_gr,
                    #     "nop": self.rgr_nop,
                    #     "p": self.rgr_p,
                    # },
                    # "RandomGridShuffle":{
                    #     "select": self.rgs_select,
                    #     "mode": self.rgs_mode,
                    #     "grid": self.rgs_grid,
                    #     "p": self.rgs_p,
                    # },
                    "RandomRain":{
                        "select": self.rr_select,
                        "mode": self.rr_mode,
                        "sl": self.rr_sl,
                        "su": self.rr_su,
                        "dl": self.rr_dl,
                        "dw": self.rr_dw,
                        "dc": self.rr_dc,
                        "bv": self.rr_bv,
                        "bc": self.rr_bc,
                        "p": self.rr_p,
                    },
                    "RandomShadow":{
                        "select": self.rsh_select,
                        "mode": self.rsh_mode,
                        "sr": self.rsh_sr,
                        "nsl": self.rsh_nsl,
                        "nsu": self.rsh_nsu,
                        "sd": self.rsh_sd,
                        "p": self.rsh_p,
                    },
                    "RandomSnow":{
                        "select": self.rsn_select,
                        "mode": self.rsn_mode,
                        "spl": self.rsn_spl,
                        "spu": self.rsn_spu,
                        "bc": self.rsn_bc,
                        "p": self.rsn_p,
                    },
                    "RandomSunFlare":{
                        "select": self.rsf_select,
                        "mode": self.rsf_mode,
                        "fr": self.rsf_fr,
                        "al": self.rsf_al,
                        "au": self.rsf_au,
                        "nfcl": self.rsf_nfcl,
                        "sc": self.rsf_sc,
                        "nfcu": self.rsf_nfcu,
                        "sr": self.rsf_sr,
                        "p": self.rsf_p,
                    },
                    "RandomToneCurve":{
                        "select": self.rtc_select,
                        "mode": self.rtc_mode,
                        "s": self.rtc_s,
                        "p": self.rtc_p,
                    },
                    "RandomCrop":{
                        "select": self.rcr_select,
                        "mode": self.rcr_mode,
                        "height": self.rcr_height,
                        "width": self.rcr_width,
                        "p": self.rcr_p,
                    },
                    "RandomRotate90":{
                        "select": self.rr9_select,
                        "mode": self.rr9_mode,
                        "p": self.rr9_p,
                    },
                    "RandomResizedCrop":{
                        "select": self.rrc_select,
                        "mode": self.rrc_mode,
                        "height": self.rrc_height,
                        "width": self.rrc_width,
                        "scale": self.rrc_scale,
                        "ratio": self.rrc_ratio,
                        "inter": self.rrc_inter,
                        "p": self.rrc_p,
                    },
                    "BBoxSafeRandomCrop":{
                        "select": self.bbsrc_select,
                        "mode": self.bbsrc_mode,
                        "er": self.bbsrc_er,
                        "p": self.bbsrc_p,
                    },
                    "RandomCropFromBorders":{
                        "select": self.rcfb_select,
                        "mode": self.rcfb_mode,
                        "cl": self.rcfb_cl,
                        "cr": self.rcfb_cr,
                        "ct": self.rcfb_ct,
                        "cb": self.rcfb_cb,
                        "p": self.rcfb_p,
                    },
                    "RandomSizedBBoxSafeCrop":{
                        "select": self.rsbbsc_select,
                        "mode": self.rsbbsc_mode,
                        "height": self.rsbbsc_height,
                        "width": self.rsbbsc_width,
                        "er": self.rsbbsc_er,
                        "inter": self.rsbbsc_inter,
                        "p": self.rsbbsc_p,
                    },
                    "RandomScale":{
                        "select": self.rs_select,
                        "mode": self.rs_mode,
                        "sl": self.rs_sl,
                        "inter": self.rs_inter,
                        "p": self.rs_p,
                    },
                    "RandomSizedCrop":{
                        "select": self.rsc_select,
                        "mode": self.rsc_mode,
                        "mmh": self.rsc_mmh,
                        "height": self.rsc_height,
                        "width": self.rsc_width,
                        "wr": self.rsc_wr,
                        "inter": self.rsc_inter,
                        "p": self.rsc_p,
                    },
                    "RandomCropNearBBox":{
                        "select": self.rcnbb_select,
                        "mode": self.rcnbb_mode,
                        "mps": self.rcnbb_mps,
                        "p": self.rcnbb_p,
                    },
                    "Crop":{
                        "select": self.crop_select,
                        "mode": self.crop_mode,
                        "xmi": self.crop_xmi,
                        "ymi": self.crop_ymi,
                        "xma": self.crop_xma,
                        "yma": self.crop_yma,
                        "p": self.crop_p,
                    },
                    "CenterCrop":{
                        "select": self.cc_select,
                        "mode": self.cc_mode,
                        "height": self.cc_height,
                        "width": self.cc_width,
                        "p": self.cc_p,
                    },
                    "CropAndPad":{
                        "select": self.cap_select,
                        "mode": self.cap_mode,
                        "px": self.cap_px,
                        "percent": self.cap_percent,
                        "pm": self.cap_pm,
                        "pc": self.cap_pc,
                        "ks": self.cap_ks,
                        "si": self.cap_si,
                        "inter": self.cap_inter,
                        "p": self.cap_p,
                    },
                    "Flip":{
                        "select": self.flip_select,
                        "mode": self.flip_mode,
                        "p": self.flip_p,
                    },
                    "Affine":{
                        "select": self.affine_select,
                        "mode": self.affine_mode,
                        "scale": self.affine_scale,
                        "tpe": self.affine_tpe,
                        "tpx": self.affine_tpx,
                        "rotate": self.affine_rotate,
                        "shear": self.affine_shear,
                        "inter": self.affine_inter,
                        "cval": self.affine_cval,
                        "mod": self.affine_mod,
                        "p": self.affine_p,
                    },
                    "Resize":{
                        "select": self.resize_select,
                        "mode": self.resize_mode,
                        "height": self.resize_height,
                        "width": self.resize_width,
                        "inter": self.resize_inter,
                        "p": self.resize_p,
                    },
                    "Rotate":{
                        "select": self.rotate_select,
                        "mode": self.rotate_mode,
                        "limit": self.rotate_limit,
                        "inter": self.rotate_inter,
                        "bm": self.rotate_bm,
                        "cb": self.rotate_cb,
                        "p": self.rotate_p,
                    },
                    "Transpose":{
                        "select": self.trans_select,
                        "mode": self.trans_mode,
                        "p": self.trans_p,
                    }
                },
                "aug_lists": str(self.aug_lists),
            }
        }
            
        # 打开文件选择框，获取文件名和路径
        fileName, _ = QFileDialog.getSaveFileName(self, '保存文件', './config/default.json', 'Json Files (*.json);;All Files (*)')

        # 如果有文件名，则保存文件
        if fileName:
            with open(fileName, 'w') as f:
                json.dump(params,f)

        
    def import_config(self):
        # 打开文件选择框，获取文件名和路径
        fileName, _ = QFileDialog.getOpenFileName(self, '打开文件', './config/default.json', 'Json Files (*.json);;All Files (*)')

        # 如果有文件名，则打开文件
        if fileName:
            with open(fileName, 'r') as f:
                p = json.load(f)
        else:
            return 
    
        self.image_path = p["aug"]["single_config"]["image_path"]
        self.anno_path = p["aug"]["single_config"]["anno_path"]
        self.dataset_format = p["aug"]["single_config"]["dataset_format"]
        self.aug_index = p["aug"]["single_config"]["index"]

        self.image_folder_path = p["aug"]["batch_config"]["image_dir"]
        self.anno_folder_path = p["aug"]["batch_config"]["anno_dir"]
        self.datasets_format = p["aug"]["batch_config"]["datasets_format"]
        self.aug_index = p["aug"]["batch_config"]["index"]            
        
        self.aug_lists = eval(p["aug"]["aug_lists"])
        
        # advancedblur
        self.advancedblur_select = p["aug"]["aug_method"]["advancedblur"]["select"]
        self.advancedblur_mode = p["aug"]["aug_method"]["advancedblur"]["mode"]
        self.advancedblur_blur = p["aug"]["aug_method"]["advancedblur"]["blur"]
        self.ab_sigmax = p["aug"]["aug_method"]["advancedblur"]["sigmax"]
        self.ab_sigmay = p["aug"]["aug_method"]["advancedblur"]["sigmay"]
        self.ab_beta_limit = p["aug"]["aug_method"]["advancedblur"]["beta_limit"]
        self.ab_noise_limit = p["aug"]["aug_method"]["advancedblur"]["noise_limit"]
        self.ab_rotate_limit = p["aug"]["aug_method"]["advancedblur"]["rotate_limit"]
        self.advancedblur_p = p["aug"]["aug_method"]["advancedblur"]["p"]

        # blur
        self.blur_select = p["aug"]["aug_method"]["blur"]["select"]
        self.blur_mode = p["aug"]["aug_method"]["blur"]["mode"]
        self.blur_blur_limit = p["aug"]["aug_method"]["blur"]["blur_limit"]
        self.blur_p = p["aug"]["aug_method"]["blur"]["p"]

        # defocus
        self.defocus_select = p["aug"]["aug_method"]["defocus"]["select"]
        self.defocus_mode = p["aug"]["aug_method"]["defocus"]["mode"]
        self.defocus_radius = p["aug"]["aug_method"]["defocus"]["radius"]
        self.defocus_alias_blur = p["aug"]["aug_method"]["defocus"]["alias_blur"]
        self.defocus_p = p["aug"]["aug_method"]["defocus"]["p"]

        # guassianblur
        self.gab_select = p["aug"]["aug_method"]["guassianblur"]["select"]
        self.gab_mode = p["aug"]["aug_method"]["guassianblur"]["mode"]
        self.gab_blur_limit = p["aug"]["aug_method"]["guassianblur"]["blur_limit"]
        self.gab_sigma_limit = p["aug"]["aug_method"]["guassianblur"]["sigma_limit"]
        self.gab_p = p["aug"]["aug_method"]["guassianblur"]["p"]

        # GlassBlur
        self.glb_select = p["aug"]["aug_method"]["GlassBlur"]["select"]
        self.glb_mode = p["aug"]["aug_method"]["GlassBlur"]["mode"]
        self.glb_sigma = p["aug"]["aug_method"]["GlassBlur"]["sigma"]
        self.glb_max_delta = p["aug"]["aug_method"]["GlassBlur"]["max_delta"]
        self.glb_iterations = p["aug"]["aug_method"]["GlassBlur"]["iterations"]
        self.glb_mode1 = p["aug"]["aug_method"]["GlassBlur"]["mode1"]
        self.glb_p = p["aug"]["aug_method"]["GlassBlur"]["p"]

        # MedianBlur
        self.meb_select = p["aug"]["aug_method"]["MedianBlur"]["select"]
        self.meb_mode = p["aug"]["aug_method"]["MedianBlur"]["mode"]
        self.meb_blur_limit = p["aug"]["aug_method"]["MedianBlur"]["blur_limit"]
        self.meb_p = p["aug"]["aug_method"]["MedianBlur"]["p"]

        # MotionBlur
        self.mob_select = p["aug"]["aug_method"]["MotionBlur"]["select"]
        self.mob_mode = p["aug"]["aug_method"]["MotionBlur"]["mode"]
        self.mob_blur_limit = p["aug"]["aug_method"]["MotionBlur"]["blur_limit"]
        self.mob_allow_shifted = p["aug"]["aug_method"]["MotionBlur"]["allow_shifted"]
        self.mob_p = p["aug"]["aug_method"]["MotionBlur"]["p"]

        # ZoomBlur
        self.zb_select = p["aug"]["aug_method"]["ZoomBlur"]["select"]
        self.zb_mode = p["aug"]["aug_method"]["ZoomBlur"]["mode"]
        self.zb_max_factor = p["aug"]["aug_method"]["ZoomBlur"]["max_factor"]
        self.zb_step_factor = p["aug"]["aug_method"]["ZoomBlur"]["step_factor"]
        self.zb_p = p["aug"]["aug_method"]["ZoomBlur"]["p"]

        # RandomBrightness
        self.rb_select = p["aug"]["aug_method"]["RandomBrightness"]["select"]
        self.rb_mode = p["aug"]["aug_method"]["RandomBrightness"]["mode"]
        self.rb_limit = p["aug"]["aug_method"]["RandomBrightness"]["limit"]
        self.rb_p = p["aug"]["aug_method"]["RandomBrightness"]["p"]

        # RandomBrightnessContrast
        self.rbc_select = p["aug"]["aug_method"]["RandomBrightnessContrast"]["select"]
        self.rbc_mode = p["aug"]["aug_method"]["RandomBrightnessContrast"]["mode"]
        self.rbc_bl = p["aug"]["aug_method"]["RandomBrightnessContrast"]["bl"]
        self.rbc_cl = p["aug"]["aug_method"]["RandomBrightnessContrast"]["cl"]
        self.rbc_bbm = p["aug"]["aug_method"]["RandomBrightnessContrast"]["bbm"]
        self.rbc_p = p["aug"]["aug_method"]["RandomBrightnessContrast"]["p"]

        # # RandomContrast
        # self.rc_select = p["aug"]["aug_method"]["RandomContrast"]["select"]
        # self.rc_mode = p["aug"]["aug_method"]["RandomContrast"]["mode"]
        # self.rc_limit = p["aug"]["aug_method"]["RandomContrast"]["limit"]
        # self.rc_p = p["aug"]["aug_method"]["RandomContrast"]["p"]

        # RandomFog 
        self.rf_select = p["aug"]["aug_method"]["RandomFog"]["select"]
        self.rf_mode = p["aug"]["aug_method"]["RandomFog"]["mode"]
        self.rf_fcl = p["aug"]["aug_method"]["RandomFog"]["fcl"]
        self.rf_fcu = p["aug"]["aug_method"]["RandomFog"]["fcu"]
        self.rf_ac = p["aug"]["aug_method"]["RandomFog"]["ac"]
        self.rf_p = p["aug"]["aug_method"]["RandomFog"]["p"]

        # RandomGamma
        self.rga_select = p["aug"]["aug_method"]["RandomGamma"]["select"]
        self.rga_mode = p["aug"]["aug_method"]["RandomGamma"]["mode"]
        self.rga_gl = p["aug"]["aug_method"]["RandomGamma"]["gl"]
        self.rga_p = p["aug"]["aug_method"]["RandomGamma"]["p"]

        # # RandomGravel
        # self.rgr_select = p["aug"]["aug_method"]["RandomGravel"]["select"]
        # self.rgr_mode = p["aug"]["aug_method"]["RandomGravel"]["mode"]
        # self.rgr_gr = p["aug"]["aug_method"]["RandomGravel"]["gr"]
        # self.rgr_nop = p["aug"]["aug_method"]["RandomGravel"]["nop"]
        # self.rgr_p = p["aug"]["aug_method"]["RandomGravel"]["p"]

        # # RandomGridShuffle
        # self.rgs_select = p["aug"]["aug_method"]["RandomGridShuffle"]["select"]
        # self.rgs_mode = p["aug"]["aug_method"]["RandomGridShuffle"]["mode"]
        # self.rgs_grid = p["aug"]["aug_method"]["RandomGridShuffle"]["grid"]
        # self.rgs_p = p["aug"]["aug_method"]["RandomGridShuffle"]["p"]

        # RandomRain
        self.rr_select = p["aug"]["aug_method"]["RandomRain"]["select"]
        self.rr_mode = p["aug"]["aug_method"]["RandomRain"]["mode"]
        self.rr_sl = p["aug"]["aug_method"]["RandomRain"]["sl"]
        self.rr_su = p["aug"]["aug_method"]["RandomRain"]["su"]
        self.rr_dl = p["aug"]["aug_method"]["RandomRain"]["dl"]
        self.rr_dw = p["aug"]["aug_method"]["RandomRain"]["dw"]
        self.rr_dc = p["aug"]["aug_method"]["RandomRain"]["dc"]
        self.rr_bv = p["aug"]["aug_method"]["RandomRain"]["bv"]
        self.rr_bc = p["aug"]["aug_method"]["RandomRain"]["bc"]
        self.rr_p = p["aug"]["aug_method"]["RandomRain"]["p"]

        # RandomShadow
        self.rsh_select = p["aug"]["aug_method"]["RandomShadow"]["select"]
        self.rsh_mode = p["aug"]["aug_method"]["RandomShadow"]["mode"]
        self.rsh_sr = p["aug"]["aug_method"]["RandomShadow"]["sr"]
        self.rsh_nsl = p["aug"]["aug_method"]["RandomShadow"]["nsl"]
        self.rsh_nsu = p["aug"]["aug_method"]["RandomShadow"]["nsu"]
        self.rsh_sd = p["aug"]["aug_method"]["RandomShadow"]["sd"]
        self.rsh_p = p["aug"]["aug_method"]["RandomShadow"]["p"]

        # RandomSnow
        self.rsn_select = p["aug"]["aug_method"]["RandomSnow"]["select"]
        self.rsn_mode = p["aug"]["aug_method"]["RandomSnow"]["mode"]
        self.rsn_spl = p["aug"]["aug_method"]["RandomSnow"]["spl"]
        self.rsn_spu = p["aug"]["aug_method"]["RandomSnow"]["spu"]
        self.rsn_bc = p["aug"]["aug_method"]["RandomSnow"]["bc"]
        self.rsn_p = p["aug"]["aug_method"]["RandomSnow"]["p"]

        # RandomSunFlare
        self.rsf_select = p["aug"]["aug_method"]["RandomSunFlare"]["select"]
        self.rsf_mode = p["aug"]["aug_method"]["RandomSunFlare"]["mode"]
        self.rsf_fr = p["aug"]["aug_method"]["RandomSunFlare"]["fr"]
        self.rsf_al = p["aug"]["aug_method"]["RandomSunFlare"]["al"]
        self.rsf_au = p["aug"]["aug_method"]["RandomSunFlare"]["au"]
        self.rsf_nfcl = p["aug"]["aug_method"]["RandomSunFlare"]["nfcl"]
        self.rsf_sc = p["aug"]["aug_method"]["RandomSunFlare"]["sc"]
        self.rsf_nfcu = p["aug"]["aug_method"]["RandomSunFlare"]["nfcu"]
        self.rsf_sr = p["aug"]["aug_method"]["RandomSunFlare"]["sr"]
        self.rsf_p = p["aug"]["aug_method"]["RandomSunFlare"]["p"]

        # RandomToneCurve
        self.rtc_select = p["aug"]["aug_method"]["RandomToneCurve"]["select"]
        self.rtc_mode = p["aug"]["aug_method"]["RandomToneCurve"]["mode"]
        self.rtc_s = p["aug"]["aug_method"]["RandomToneCurve"]["s"]
        self.rtc_p = p["aug"]["aug_method"]["RandomToneCurve"]["p"]

        # RandomCrop
        self.rcr_select = p["aug"]["aug_method"]["RandomCrop"]["select"]
        self.rcr_mode = p["aug"]["aug_method"]["RandomCrop"]["mode"]
        self.rcr_height = p["aug"]["aug_method"]["RandomCrop"]["height"]
        self.rcr_width = p["aug"]["aug_method"]["RandomCrop"]["width"]
        self.rcr_p = p["aug"]["aug_method"]["RandomCrop"]["p"]

        # RandomRotate90
        self.rr9_select = p["aug"]["aug_method"]["RandomRotate90"]["select"]
        self.rr9_mode = p["aug"]["aug_method"]["RandomRotate90"]["mode"]
        self.rr9_p = p["aug"]["aug_method"]["RandomRotate90"]["p"]

        # RandomResizedCrop
        self.rrc_select = p["aug"]["aug_method"]["RandomResizedCrop"]["select"]
        self.rrc_mode = p["aug"]["aug_method"]["RandomResizedCrop"]["mode"]
        self.rrc_height = p["aug"]["aug_method"]["RandomResizedCrop"]["height"]
        self.rrc_width = p["aug"]["aug_method"]["RandomResizedCrop"]["width"]
        self.rrc_scale = p["aug"]["aug_method"]["RandomResizedCrop"]["scale"]
        self.rrc_ratio = p["aug"]["aug_method"]["RandomResizedCrop"]["ratio"]
        self.rrc_inter = p["aug"]["aug_method"]["RandomResizedCrop"]["inter"]
        self.rrc_p = p["aug"]["aug_method"]["RandomResizedCrop"]["p"]

        # BBoxSafeRandomCrop
        self.bbsrc_select = p["aug"]["aug_method"]["BBoxSafeRandomCrop"]["select"]
        self.bbsrc_mode = p["aug"]["aug_method"]["BBoxSafeRandomCrop"]["mode"]
        self.bbsrc_er = p["aug"]["aug_method"]["BBoxSafeRandomCrop"]["er"]
        self.bbsrc_p = p["aug"]["aug_method"]["BBoxSafeRandomCrop"]["p"]

        # RandomCropFromBorders
        self.rcfb_select = p["aug"]["aug_method"]["RandomCropFromBorders"]["select"]
        self.rcfb_mode = p["aug"]["aug_method"]["RandomCropFromBorders"]["mode"]
        self.rcfb_cl = p["aug"]["aug_method"]["RandomCropFromBorders"]["cl"]
        self.rcfb_cr = p["aug"]["aug_method"]["RandomCropFromBorders"]["cr"]
        self.rcfb_ct = p["aug"]["aug_method"]["RandomCropFromBorders"]["ct"]
        self.rcfb_cb = p["aug"]["aug_method"]["RandomCropFromBorders"]["cb"]
        self.rcfb_p = p["aug"]["aug_method"]["RandomCropFromBorders"]["p"]

        # RandomSizedBBoxSafeCrop	
        self.rsbbsc_select = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["select"]
        self.rsbbsc_mode = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["mode"]
        self.rsbbsc_height = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["height"]
        self.rsbbsc_width = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["width"]
        self.rsbbsc_er = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["er"]
        self.rsbbsc_inter = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["inter"]
        self.rsbbsc_p = p["aug"]["aug_method"]["RandomSizedBBoxSafeCrop"]["p"]

        # RandomScale
        self.rs_select = p["aug"]["aug_method"]["RandomScale"]["select"]
        self.rs_mode = p["aug"]["aug_method"]["RandomScale"]["mode"]
        self.rs_sl = p["aug"]["aug_method"]["RandomScale"]["sl"]
        self.rs_inter = p["aug"]["aug_method"]["RandomScale"]["inter"]
        self.rs_p = p["aug"]["aug_method"]["RandomScale"]["p"]

        # RandomSizedCrop
        self.rsc_select = p["aug"]["aug_method"]["RandomSizedCrop"]["select"]
        self.rsc_mode = p["aug"]["aug_method"]["RandomSizedCrop"]["mode"]
        self.rsc_mmh = p["aug"]["aug_method"]["RandomSizedCrop"]["mmh"]
        self.rsc_height = p["aug"]["aug_method"]["RandomSizedCrop"]["height"]
        self.rsc_width = p["aug"]["aug_method"]["RandomSizedCrop"]["width"]
        self.rsc_wr = p["aug"]["aug_method"]["RandomSizedCrop"]["wr"]
        self.rsc_inter = p["aug"]["aug_method"]["RandomSizedCrop"]["inter"]
        self.rsc_p = p["aug"]["aug_method"]["RandomSizedCrop"]["p"]

        # RandomCropNearBBox
        self.rcnbb_select = p["aug"]["aug_method"]["RandomCropNearBBox"]["select"]
        self.rcnbb_mode = p["aug"]["aug_method"]["RandomCropNearBBox"]["mode"]
        self.rcnbb_mps = p["aug"]["aug_method"]["RandomCropNearBBox"]["mps"]
        self.rcnbb_p = p["aug"]["aug_method"]["RandomCropNearBBox"]["p"]

        # Crop
        self.crop_select = p["aug"]["aug_method"]["Crop"]["select"]
        self.crop_mode = p["aug"]["aug_method"]["Crop"]["mode"]
        self.crop_xmi = p["aug"]["aug_method"]["Crop"]["xmi"]
        self.crop_ymi = p["aug"]["aug_method"]["Crop"]["ymi"]
        self.crop_xma = p["aug"]["aug_method"]["Crop"]["xma"]
        self.crop_yma = p["aug"]["aug_method"]["Crop"]["yma"]
        self.crop_p = p["aug"]["aug_method"]["Crop"]["p"]

        # CenterCrop
        self.cc_select = p["aug"]["aug_method"]["CenterCrop"]["select"]
        self.cc_mode = p["aug"]["aug_method"]["CenterCrop"]["mode"]
        self.cc_height = p["aug"]["aug_method"]["CenterCrop"]["height"]
        self.cc_width = p["aug"]["aug_method"]["CenterCrop"]["width"]
        self.cc_p = p["aug"]["aug_method"]["CenterCrop"]["p"]

        # CropAndPad
        self.cap_select = p["aug"]["aug_method"]["CropAndPad"]["select"]
        self.cap_mode = p["aug"]["aug_method"]["CropAndPad"]["mode"]
        self.cap_px = p["aug"]["aug_method"]["CropAndPad"]["px"]
        self.cap_percent = p["aug"]["aug_method"]["CropAndPad"]["percent"]
        self.cap_pm = p["aug"]["aug_method"]["CropAndPad"]["pm"]
        self.cap_pc = p["aug"]["aug_method"]["CropAndPad"]["pc"]
        self.cap_ks = p["aug"]["aug_method"]["CropAndPad"]["ks"]
        self.cap_si = p["aug"]["aug_method"]["CropAndPad"]["si"]
        self.cap_inter = p["aug"]["aug_method"]["CropAndPad"]["inter"]
        self.cap_p = p["aug"]["aug_method"]["CropAndPad"]["p"]

        # Flip
        self.flip_select = p["aug"]["aug_method"]["Flip"]["select"]
        self.flip_mode = p["aug"]["aug_method"]["Flip"]["mode"]
        self.flip_p = p["aug"]["aug_method"]["Flip"]["p"]

        # Affine
        self.affine_select = p["aug"]["aug_method"]["Affine"]["select"]
        self.affine_mode = p["aug"]["aug_method"]["Affine"]["mode"]
        self.affine_scale = p["aug"]["aug_method"]["Affine"]["scale"]
        self.affine_tpe = p["aug"]["aug_method"]["Affine"]["tpe"]
        self.affine_tpx = p["aug"]["aug_method"]["Affine"]["tpx"]
        self.affine_rotate = p["aug"]["aug_method"]["Affine"]["rotate"]
        self.affine_shear = p["aug"]["aug_method"]["Affine"]["shear"]
        self.affine_inter = p["aug"]["aug_method"]["Affine"]["inter"]
        self.affine_cval = p["aug"]["aug_method"]["Affine"]["cval"]
        self.affine_mod = p["aug"]["aug_method"]["Affine"]["mod"]
        self.affine_p = p["aug"]["aug_method"]["Affine"]["p"]

        # Resize
        self.resize_select = p["aug"]["aug_method"]["Resize"]["select"]
        self.resize_mode = p["aug"]["aug_method"]["Resize"]["mode"]
        self.resize_height = p["aug"]["aug_method"]["Resize"]["height"]
        self.resize_width = p["aug"]["aug_method"]["Resize"]["width"]
        self.resize_inter = p["aug"]["aug_method"]["Resize"]["inter"]
        self.resize_p = p["aug"]["aug_method"]["Resize"]["p"]

        # Rotate
        self.rotate_select = p["aug"]["aug_method"]["Rotate"]["select"]
        self.rotate_mode = p["aug"]["aug_method"]["Rotate"]["mode"]
        self.rotate_limit = p["aug"]["aug_method"]["Rotate"]["limit"]
        self.rotate_inter = p["aug"]["aug_method"]["Rotate"]["inter"]
        self.rotate_bm = p["aug"]["aug_method"]["Rotate"]["bm"]
        self.rotate_cb = p["aug"]["aug_method"]["Rotate"]["cb"]
        self.rotate_p = p["aug"]["aug_method"]["Rotate"]["p"]

        # Transpose
        self.trans_select = p["aug"]["aug_method"]["Transpose"]["select"]
        self.trans_mode = p["aug"]["aug_method"]["Transpose"]["mode"]
        self.trans_p = p["aug"]["aug_method"]["Transpose"]["p"]
        # self.aug_lists = []
        self.setVar()
        # self.add_method()
        self.apply_image()


    # 变量初始化，可用于参数配置导入导出
    def initVar(self):
        self.cm_initVar()
        self.ia_initVar()
        self.mi_initVar()
        self.image_path = os.path.join(os.getcwd(),"data","dataset_single","demo.jpg")
        
        self.anno_path = os.path.join(os.getcwd(),"data","dataset_single","demo.txt")
        self.dataset_format = 0
        
        # 批量配置
        self.image_folder_path = os.path.join(os.getcwd(),"data","dataset_batch","images")
        self.anno_folder_path = os.path.join(os.getcwd(),"data","dataset_batch","labels")
        self.datasets_format = 0
        self.aug_output_dir = os.path.join(os.getcwd(),"data","output_aug")
        
        self.aug_index = 0
        
        # 增强列表
        self.aug_lists = []
        # advancedblur
        self.advancedblur_select = 0
        # mode = ["Compose","Oneof"]
        self.advancedblur_mode = 1
        self.advancedblur_blur=[3,7]
        self.ab_sigmax = [0.2,1.0]
        self.ab_sigmay = [0.2,1.0]
        self.ab_beta_limit = [0.5,8.0]
        self.ab_noise_limit = [0.9,1.1]
        self.ab_rotate_limit = 90
        self.advancedblur_p = 0.5
        # blur
        self.blur_select = 0
        self.blur_mode = 1
        self.blur_blur_limit = [2,3]
        self.blur_p = 0.5
        # defocus
        self.defocus_select = 0
        self.defocus_mode = 1
        self.defocus_radius = [3,10]
        self.defocus_alias_blur = [0.1,0.5]
        self.defocus_p = 0.5
        # guassianblur
        self.gab_select = 0
        self.gab_mode = 1
        self.gab_blur_limit = [3,7]
        self.gab_sigma_limit = [0.0,0.0]
        self.gab_p = 0.5
        # GlassBlur
        self.glb_select = 0
        self.glb_mode = 1
        self.glb_sigma = 0.3
        self.glb_max_delta = 1
        self.glb_iterations = 2
        # ['fast','exact']
        self.glb_mode1 = 0
        self.glb_p = 0.5

        # MedianBlur
        self.meb_select = 0
        self.meb_mode = 1
        self.meb_blur_limit=7
        self.meb_p = 0.5
        # MotionBlur
        self.mob_select = 0
        self.mob_mode = 0
        self.mob_blur_limit=7
        self.mob_allow_shifted = 1
        self.mob_p = 0.5
        # ZoomBlur
        self.zb_select = 0
        self.zb_mode = 1
        self.zb_max_factor=[1.00,1.05]
        self.zb_step_factor=[0.01,0.02]
        self.zb_p = 0.5

        # RandomBrightness
        self.rb_select = 0
        self.rb_mode = 0
        self.rb_limit = [-0.2,0.2]
        self.rb_p = 0.5
        # RandomBrightnessContrast
        self.rbc_select = 0
        self.rbc_mode = 0
        self.rbc_bl = [-0.4,0.2]
        self.rbc_cl = [-0.2,0.2]
        self.rbc_bbm = 1
        self.rbc_p = 0.5 
        # # RandomContrast
        # self.rc_select = 0
        # self.rc_mode = 0
        # self.rc_limit = [-0.2,0.2]
        # self.rc_p = 0.6
        # RandomFog 
        self.rf_select = 0
        self.rf_mode = 1
        self.rf_fcl = 0.0
        self.rf_fcu = 0.3
        self.rf_ac = 0.08
        self.rf_p = 0.5
        # RandomGamma
        self.rga_select = 0
        self.rga_mode = 0
        self.rga_gl = [50.0,150.0]
        self.rga_p = 0.5
        # # RandomGravel
        # self.rgr_select = 0
        # self.rgr_mode = 0
        # self.rgr_gr = [0.1,0.4,0.9,0.9]
        # self.rgr_nop = 0 
        # self.rgr_p = 0.5
        # # RandomGridShuffle
        # self.rgs_select = 0
        # self.rgs_mode = 0
        # self.rgs_grid = [3,3]
        # self.rgs_p = 0.5
        # RandomRain
        self.rr_select = 0
        self.rr_mode = 1
        self.rr_sl = -10
        self.rr_su = 10
        self.rr_dl = 10
        self.rr_dw = 1
        self.rr_dc = [200,200,200]
        self.rr_bv = 7
        self.rr_bc = 0.8
        self.rr_p = 0.5
        # RandomShadow
        self.rsh_select = 0
        self.rsh_mode = 1
        self.rsh_sr = [0,0,1,1]
        self.rsh_nsl = 1
        self.rsh_nsu = 2
        self.rsh_sd = 3
        self.rsh_p = 0.5
        # RandomSnow
        self.rsn_select = 0
        self.rsn_mode = 1
        self.rsn_spl = 0
        self.rsn_spu = 0.2
        self.rsn_bc = 2.5
        self.rsn_p = 0.5
        # RandomSunFlare
        self.rsf_select = 0 
        self.rsf_mode = 1
        self.rsf_fr = [0,0,0.1,0.5]
        self.rsf_al = 0.0
        self.rsf_au = 1.0
        self.rsf_nfcl = 6
        self.rsf_sc = [255,255,255]
        self.rsf_nfcu = 10
        self.rsf_sr = 400
        self.rsf_p = 0.5
        # RandomToneCurve
        self.rtc_select = 0
        self.rtc_mode = 0
        self.rtc_s = 0.1
        self.rtc_p = 0.5
        
        # RandomCrop
        self.rcr_select = 0
        self.rcr_mode = 0
        self.rcr_height = 640
        self.rcr_width = 640
        self.rcr_p = 0.5
        # RandomRotate90
        self.rr9_select = 0
        self.rr9_mode = 0
        self.rr9_p = 0.5
        # RandomResizedCrop
        self.rrc_select = 0
        self.rrc_mode = 0
        self.rrc_height = 640
        self.rrc_width = 640
        self.rrc_scale = [0.08,1.0]
        self.rrc_ratio = [0.75,1.33]
        self.rrc_inter = 1
        self.rrc_p = 0.5
        # BBoxSafeRandomCrop
        self.bbsrc_select = 0
        self.bbsrc_mode = 0
        self.bbsrc_er = 0.05
        self.bbsrc_p = 0.5
        # RandomCropFromBorders
        self.rcfb_select = 0
        self.rcfb_mode = 0
        self.rcfb_cl = 0.1
        self.rcfb_cr = 0.1
        self.rcfb_ct = 0.1
        self.rcfb_cb = 0.1
        self.rcfb_p = 0.5
        # RandomSizedBBoxSafeCrop	
        self.rsbbsc_select = 0
        self.rsbbsc_mode = 0
        self.rsbbsc_height = 640
        self.rsbbsc_width = 640
        self.rsbbsc_er = 0.0
        self.rsbbsc_inter = 1
        self.rsbbsc_p = 0.5
        # RandomScale
        self.rs_select = 0
        self.rs_mode = 0
        self.rs_sl = [-0.2,0.2]
        self.rs_inter = 1
        self.rs_p = 0.5
        # RandomSizedCrop
        self.rsc_select = 0
        self.rsc_mode = 0
        self.rsc_mmh = [0,1000]
        self.rsc_height = 640
        self.rsc_width = 640
        self.rsc_wr = 1.0
        self.rsc_inter = 1
        self.rsc_p = 0.5
        # RandomCropNearBBox
        self.rcnbb_select = 0
        self.rcnbb_mode = 0
        self.rcnbb_mps = [0.0,0.0]
        self.rcnbb_p = 0.5


        # Crop
        self.crop_select = 0
        self.crop_mode = 0
        self.crop_xmi = 0
        self.crop_ymi = 0
        self.crop_xma = 1024
        self.crop_yma = 1024
        self.crop_p = 0.5
        # CenterCrop
        self.cc_select = 0
        self.cc_mode = 0
        self.cc_height = 640
        self.cc_width = 640
        self.cc_p = 0.5
        # CropAndPad
        self.cap_select = 0
        self.cap_mode = 0
        self.cap_px = 50
        self.cap_percent = 0.0
        self.cap_pm = 0
        self.cap_pc = 0
        self.cap_ks = 1
        self.cap_si = 1
        self.cap_inter = 1
        self.cap_p = 0.5
        # Flip
        self.flip_select = 0
        self.flip_mode = 0
        self.flip_p = 0.5
        # Affine
        self.affine_select = 0
        self.affine_mode = 0
        self.affine_scale = [0.0,0.0]
        self.affine_tpe = [0.0,0.0]
        self.affine_tpx = [0,0]
        self.affine_rotate = [0,0]
        self.affine_shear = [0.0,0.0]
        self.affine_inter = 0
        self.affine_cval = 0
        self.affine_mod = 0
        self.affine_p = 0.3
        # Resize
        self.resize_select = 0
        self.resize_mode = 0
        self.resize_height = 640
        self.resize_width = 640
        self.resize_inter = 1
        self.resize_p = 0.5
        # Rotate
        self.rotate_select = 0
        self.rotate_mode = 0
        self.rotate_limit = [-90,90]
        self.rotate_inter = 1
        self.rotate_bm = 0
        self.rotate_cb = 0
        self.rotate_p = 0.5
        # Transpose
        self.trans_select = 0
        self.trans_mode = 0
        self.trans_p = 0.5

        # 数据处理
        ## easydl
        self.cookie = "'请输入cookie或设置cookie.txt文件'"
        self.dataset_id = "1754956"
        self.saved_path = os.path.join(os.getcwd(),"data",'dataset_easydl')
        self.easydl_log = ""
        ## 数据转换
        self.trans_image_dir =os.path.join(os.getcwd(),"data","dataset_easydl")
        self.trans_anno_dir = os.path.join(os.getcwd(),"data","dataset_easydl")
        self.trans_label = "yyzz yz"
        self.trans_src_format = 2
        self.trans_dst_format = 0
        self.trans_output_dir = os.path.join(os.getcwd(),"data",'output_easydl')
        
    def setVar(self):
        self.cm_setVar()
        self.ia_setVar()
        self.mi_setVar()
        # 单图配置
        self.line_edit_image_path.setText(self.image_path)
        self.line_edit_anno_path.setText(self.anno_path)
        self.updateUI()
        self.show_src_image()
        self.sb_aug_index.setValue(self.aug_index)
        
        # 批量配置
        self.line_edit_image_folder_path.setText(self.image_folder_path)
        self.line_edit_anno_folder_path.setText(self.anno_folder_path)
        self.le_aug_output_dir.setText(self.aug_output_dir)

        
        # 数据格式
        self.cob_dataset_format.setCurrentIndex(self.DATASE_FORMAT[self.dataset_format] if isinstance(self.dataset_format,str) else self.dataset_format)
        self.cob_datasets_format.setCurrentIndex(self.DATASE_FORMAT[self.datasets_format] if isinstance(self.datasets_format,str) else self.datasets_format)
        
        if len(self.aug_lists)>0:
            self.pte_aug_method.setPlainText("["+",\n".join([str(i) for i in self.aug_lists])+"]")
        else:
            self.pte_aug_method.setPlainText("")
        # 数据增强
        # advancedblur
        self.checkbox_advancedblur.setChecked(self.advancedblur_select)
        self.cb_advancedblur_mode.setCurrentIndex(self.MODE[self.advancedblur_mode] if isinstance(self.advancedblur_mode,str) else self.advancedblur_mode)
        self.sb_advancedblur_blur_l.setValue(self.advancedblur_blur[0])
        self.sb_advancedblur_blur_h.setValue(self.advancedblur_blur[1])
        self.sb_ab_sigmax_l.setValue(self.ab_sigmax[0])
        self.sb_ab_sigmax_h.setValue(self.ab_sigmax[1])
        self.sb_ab_sigmay_l.setValue(self.ab_sigmay[0])
        self.sb_ab_sigmay_h.setValue(self.ab_sigmay[1])
        self.sb_ab_beta_limit_l.setValue(self.ab_beta_limit[0])
        self.sb_ab_beta_limit_h.setValue(self.ab_beta_limit[1])
        self.le_ab_noise_limit_l.setValue(self.ab_noise_limit[0])
        self.le_ab_noise_limit_h.setValue(self.ab_noise_limit[1])
        self.le_ab_rotate_limit.setText(str(self.ab_rotate_limit))
        self.sb_advancedblur_p.setValue(self.advancedblur_p)
        
        # blur
        self.cb_blur_select.setChecked(self.blur_select)
        self.cb_blur_mode.setCurrentIndex(self.MODE[self.blur_mode] if isinstance(self.blur_mode,str) else self.blur_mode)
        self.sb_blur_blur_limit0.setValue(self.blur_blur_limit[0])
        self.sb_blur_blur_limit1.setValue(self.blur_blur_limit[1])
        self.sb_blur_p.setValue(self.blur_p)
        
        # Defocus
        self.chb_defocus_select.setChecked(self.defocus_select)
        self.cob_defocus_mode.setCurrentIndex(self.MODE[self.defocus_mode] if isinstance(self.defocus_mode,str) else self.defocus_mode)
        self.sb_defocus_radius0.setValue(self.defocus_radius[0])
        self.sb_defocus_radius1.setValue(self.defocus_radius[1])
        self.sb_defocus_alias_blur0.setValue(self.defocus_alias_blur[0])
        self.sb_defocus_alias_blur1.setValue(self.defocus_alias_blur[1])
        self.sb_defocus_p.setValue(self.defocus_p)
        # guassianblur
        self.chb_gab_select.setChecked(self.gab_select)
        self.cob_gab_mode.setCurrentIndex(self.MODE[self.gab_mode] if isinstance(self.gab_mode,str) else self.gab_mode)
        self.sb_gab_blur_limit0.setValue(self.gab_blur_limit[0])
        self.sb_gab_blur_limit1.setValue(self.gab_blur_limit[1])
        self.sb_gab_sigma_limit0.setValue(self.gab_sigma_limit[0])
        self.sb_gab_sigma_limit1.setValue(self.gab_sigma_limit[1])
        self.sb_gab_p.setValue(self.gab_p)
        # GlassBlur
        self.chb_glb_select.setChecked(self.glb_select)
        self.cob_glb_mode.setCurrentIndex(self.MODE[self.glb_mode] if isinstance(self.glb_mode,str) else self.glb_mode)
        self.sb_glb_sigma.setValue(self.glb_sigma)
        self.sb_glb_max_delta.setValue(self.glb_max_delta)
        self.sb_glb_iterations.setValue(self.glb_iterations)
        self.cob_glb_mode1.setCurrentIndex(self.MODE1[self.glb_mode1] if isinstance(self.glb_mode1,str) else self.glb_mode1)
        self.sb_glb_p.setValue(self.glb_p)
        # MedianBlur
        self.chb_meb_select.setChecked(self.meb_select)
        self.cob_meb_mode.setCurrentIndex(self.MODE[self.meb_mode] if isinstance(self.meb_mode,str) else self.meb_mode)
        self.sb_meb_blur_limit.setValue(self.meb_blur_limit)
        self.sb_meb_p.setValue(self.meb_p)
        # MotionBlur
        self.chb_mob_select.setChecked(self.mob_select)
        self.cob_mob_mode.setCurrentIndex(self.MODE[self.mob_mode] if isinstance(self.mob_mode,str) else self.mob_mode)
        self.sb_mob_blur_limit.setValue(self.mob_blur_limit)
        self.chb_mob_allow_shifted.setChecked(self.mob_allow_shifted)
        self.sb_mob_p.setValue(self.meb_p)
        # ZoomBlur
        self.chb_zb_select.setChecked(self.zb_select)
        self.cob_zb_mode.setCurrentIndex(self.MODE[self.zb_mode] if isinstance(self.zb_mode,str) else self.zb_mode)
        self.sb_zb_max_factor0.setValue(self.zb_max_factor[0])
        self.sb_zb_max_factor1.setValue(self.zb_max_factor[1])
        self.sb_zb_step_factor0.setValue(self.zb_step_factor[0])
        self.sb_zb_step_factor1.setValue(self.zb_step_factor[1])
        self.sb_zb_p.setValue(self.zb_p)
        
        # RandomBrightness
        self.chb_rb_select.setChecked(self.rb_select)
        self.cob_rb_mode.setCurrentIndex(self.MODE[self.rb_mode] if isinstance(self.rb_mode,str) else self.rb_mode)
        self.sb_rb_limit0.setValue(self.rb_limit[0])
        self.sb_rb_limit1.setValue(self.rb_limit[1])
        self.sb_rb_p.setValue(self.rb_p)
        # RandomBrightnessContrast
        self.chb_rbc_select.setChecked(self.rbc_select)
        self.cob_rbc_mode.setCurrentIndex(self.MODE[self.rbc_mode] if isinstance(self.rbc_mode,str) else self.rbc_mode)
        self.sb_rbc_bl0.setValue(self.rbc_bl[0])
        self.sb_rbc_bl1.setValue(self.rbc_bl[1])
        self.sb_rbc_cl0.setValue(self.rbc_cl[0])
        self.sb_rbc_cl1.setValue(self.rbc_cl[1])
        self.chb_rbc_bbm.setChecked(self.rbc_bbm)
        self.sb_rbc_p.setValue(self.rbc_p)
        # # RandomContrast
        # self.chb_rc_select.setChecked(self.rc_select)
        # self.cob_rc_mode.setCurrentIndex(self.MODE[self.rc_mode] if isinstance(self.rc_mode,str) else self.rc_mode)
        # self.sb_rc_limit0.setValue(self.rc_limit[0])
        # self.sb_rc_limit1.setValue(self.rc_limit[1])
        # self.sb_rc_p.setValue(self.rc_p)
        # RandomFog 
        self.chb_rf_select.setChecked(self.rf_select)
        self.cob_rf_mode.setCurrentIndex(self.MODE[self.rf_mode] if isinstance(self.rf_mode,str) else self.rf_mode)
        self.sb_rf_fcl.setValue(self.rf_fcl)
        self.sb_rf_fcu.setValue(self.rf_fcu)
        self.sb_rf_ac.setValue(self.rf_ac)
        self.sb_rf_p.setValue(self.rf_p)
        # RandomGamma
        self.chb_rga_select.setChecked(self.rga_select)
        self.cob_rga_mode.setCurrentIndex(self.MODE[self.rga_mode] if isinstance(self.rga_mode,str) else self.rga_mode)
        self.sb_rga_gl0.setValue(self.rga_gl[0])
        self.sb_rga_gl1.setValue(self.rga_gl[1])
        self.sb_rga_p.setValue(self.rga_p)
        # # RandomGravel
        # self.chb_rgr_select.setChecked(self.rgr_select)
        # self.cob_rgr_mode.setCurrentIndex(self.MODE[self.rgr_mode] if isinstance(self.rgr_mode,str) else self.rgr_mode)
        # self.sb_rgr_gr0.setValue(self.rgr_gr[0])
        # self.sb_rgr_gr1.setValue(self.rgr_gr[1])
        # self.sb_rgr_gr2.setValue(self.rgr_gr[2])
        # self.sb_rgr_gr3.setValue(self.rgr_gr[3])
        # self.sb_rgr_nop.setValue(self.rgr_nop)
        # self.sb_rgr_p.setValue(self.rgr_p)
        # # RandomGridShuffle
        # self.chb_rgs_select.setChecked(self.rgs_select)
        # self.cob_rgs_mode.setCurrentIndex(self.MODE[self.rgs_mode] if isinstance(self.rgs_mode,str) else self.rgs_mode)
        # self.sb_rgs_grid0.setValue(self.rgs_grid[0])
        # self.sb_rgs_grid1.setValue(self.rgs_grid[1])
        # self.sb_rgs_p.setValue(self.rgs_p)
        # RandomRain
        self.chb_rr_select.setChecked(self.rr_select)
        self.cob_rr_mode.setCurrentIndex(self.MODE[self.rr_mode] if isinstance(self.rr_mode,str) else self.rr_mode)
        self.sb_rr_sl.setValue(self.rr_sl)
        self.sb_rr_su.setValue(self.rr_su)
        self.sb_rr_dl.setValue(self.rr_dl)
        self.sb_rr_dw.setValue(self.rr_dw)
        self.sb_rr_dc0.setValue(self.rr_dc[0])
        self.sb_rr_dc1.setValue(self.rr_dc[1])
        self.sb_rr_bv.setValue(self.rr_bv)
        self.sb_rr_bc.setValue(self.rr_bc)
        self.sb_rr_p.setValue(self.rr_p)
        # RandomShadow
        self.chb_rsh_select.setChecked(self.rsh_select)
        self.cob_rsh_mode.setCurrentIndex(self.MODE[self.rsh_mode] if isinstance(self.rsh_mode,str) else self.rsh_mode)
        self.sb_rsh_sr0.setValue(self.rsh_sr[0])
        self.sb_rsh_sr1.setValue(self.rsh_sr[1])
        self.sb_rsh_sr2.setValue(self.rsh_sr[2])
        self.sb_rsh_sr3.setValue(self.rsh_sr[3])
        self.sb_rsh_nsl.setValue(self.rsh_nsl)
        self.sb_rsh_nsu.setValue(self.rsh_nsu)
        self.sb_rsh_sd.setValue(self.rsh_sd)
        self.sb_rsh_p.setValue(self.rsh_p)
        # RandomSnow
        self.chb_rsn_select.setChecked(self.rsn_select)
        self.cob_rsn_mode.setCurrentIndex(self.MODE[self.rsn_mode] if isinstance(self.rsn_mode,str) else self.rsn_mode)
        self.sb_rsn_spl.setValue(self.rsn_spl)
        self.sb_rsn_spu.setValue(self.rsn_spu)
        self.sb_rsn_bc.setValue(self.rsn_bc)
        self.sb_rsn_p.setValue(self.rsn_p)
        # RandomSunFlare
        self.chb_rsf_select.setChecked(self.rsf_select)
        self.cob_rsf_mode.setCurrentIndex(self.MODE[self.rsf_mode] if isinstance(self.rsf_mode,str) else self.rsf_mode)
        self.sb_rsf_fr0.setValue(self.rsf_fr[0])
        self.sb_rsf_fr1.setValue(self.rsf_fr[1])
        self.sb_rsf_fr2.setValue(self.rsf_fr[2])
        self.sb_rsf_fr3.setValue(self.rsf_fr[3])
        self.sb_rsf_al.setValue(self.rsf_al)
        self.sb_rsf_au.setValue(self.rsf_au)
        self.sb_rsf_nfcl.setValue(self.rsf_nfcl)
        self.sb_rsf_sc0.setValue(self.rsf_sc[0])
        self.sb_rsf_sc1.setValue(self.rsf_sc[1])
        self.sb_rsf_sc2.setValue(self.rsf_sc[2])
        self.sb_rsf_nfcu.setValue(self.rsf_nfcu)
        self.sb_rsf_sr.setValue(self.rsf_sr)
        self.sb_rsf_p.setValue(self.rsf_p)
        # RandomToneCurve
        self.chb_rtc_select.setChecked(self.rtc_select)
        self.cob_rtc_mode.setCurrentIndex(self.MODE[self.rtc_mode] if isinstance(self.rtc_mode,str) else self.rtc_mode)
        self.sb_rtc_s.setValue(self.rtc_s)
        self.sb_rtc_p.setValue(self.rtc_p)
        
        # Spatial
        # RandomCrop
        self.chb_rcr_select.setChecked(self.rcr_select)
        self.cob_rcr_mode.setCurrentIndex(self.MODE[self.rcr_mode] if isinstance(self.rcr_mode,str) else self.rcr_mode)
        self.sb_rcr_height.setValue(self.rcr_height)
        self.sb_rcr_width.setValue(self.rcr_width)
        self.sb_rcr_p.setValue(self.rcr_p)
        # RandomRotate90
        self.chb_rr9_select.setChecked(self.rr9_select)
        self.cob_rr9_mode.setCurrentIndex(self.MODE[self.rr9_mode] if isinstance(self.rr9_mode,str) else self.rr9_mode)
        self.sb_rr9_p.setValue(self.rr9_p)
        # RandomResizedCrop
        self.chb_rrc_select.setChecked(self.rrc_select)
        self.cob_rrc_mode.setCurrentIndex(self.MODE[self.rrc_mode] if isinstance(self.rrc_mode,str) else self.rrc_mode)
        self.sb_rrc_height.setValue(self.rrc_height)
        self.sb_rrc_width.setValue(self.rrc_width)
        self.sb_rrc_scale0.setValue(self.rrc_scale[0])
        self.sb_rrc_scale1.setValue(self.rrc_scale[1])
        self.sb_rrc_ratio0.setValue(self.rrc_ratio[0])
        self.sb_rrc_ratio1.setValue(self.rrc_ratio[1])
        self.cob_rrc_inter.setCurrentIndex(self.rrc_inter)
        self.sb_rrc_p.setValue(self.rrc_p)
        # BBoxSafeRandomCrop
        self.chb_bbsrc_select.setChecked(self.bbsrc_select)
        self.cob_bbsrc_mode.setCurrentIndex(self.MODE[self.bbsrc_mode] if isinstance(self.bbsrc_mode,str) else self.bbsrc_mode)
        self.sb_bbsrc_er.setValue(self.bbsrc_er)
        self.sb_bbsrc_p.setValue(self.bbsrc_p)
        # RandomCropFromBorders
        self.chb_rcfb_select.setChecked(self.rcfb_select)
        self.cob_rcfb_mode.setCurrentIndex(self.MODE[self.rcfb_mode] if isinstance(self.rcfb_mode,str) else self.rcfb_mode)
        self.sb_rcfb_cl.setValue(self.rcfb_cl)
        self.sb_rcfb_cr.setValue(self.rcfb_cr)
        self.sb_rcfb_ct.setValue(self.rcfb_ct)
        self.sb_rcfb_cb.setValue(self.rcfb_cb)
        self.sb_rcfb_p.setValue(self.rcfb_p)
        # RandomSizedBBoxSafeCrop	
        self.chb_rsbbsc_select.setChecked(self.rsbbsc_select)
        self.cob_rsbbsc_mode.setCurrentIndex(self.MODE[self.rsbbsc_mode] if isinstance(self.rsbbsc_mode,str) else self.rsbbsc_mode)
        self.sb_rsbbsc_height.setValue(self.rsbbsc_height)
        self.sb_rsbbsc_width.setValue(self.rsbbsc_width)
        self.sb_rsbbsc_er.setValue(self.rsbbsc_er)
        self.cob_rsbbsc_inter.setCurrentIndex(self.interpolation[self.rsbbsc_inter] if isinstance(self.rsbbsc_inter,str) else self.rsbbsc_inter)
        self.sb_rsbbsc_p.setValue(self.rsbbsc_p)
        # RandomScale
        self.chb_rs_select.setChecked(self.rs_select)
        self.cob_rs_mode.setCurrentIndex(self.MODE[self.rs_mode] if isinstance(self.rs_mode,str) else self.rs_mode)
        self.sb_rs_sl0.setValue(self.rs_sl[0])
        self.sb_rs_sl1.setValue(self.rs_sl[1])
        self.cob_rs_inter.setCurrentIndex(self.interpolation[self.rs_inter] if isinstance(self.rs_inter,str) else self.rs_inter)
        self.sb_rs_p.setValue(self.rs_p)
        # RandomSizedCrop
        self.chb_rsc_select.setChecked(self.rsc_select)
        self.cob_rsc_mode.setCurrentIndex(self.MODE[self.rsc_mode] if isinstance(self.rsc_mode,str) else self.rsc_mode)
        self.sb_rsc_mmh0.setValue(self.rsc_mmh[0])
        self.sb_rsc_mmh1.setValue(self.rsc_mmh[1])
        self.sb_rsc_height.setValue(self.rsc_height)
        self.sb_rsc_width.setValue(self.rsc_width)
        self.sb_rsc_wr.setValue(self.rsc_wr)
        self.cob_rsc_inter.setCurrentIndex(self.interpolation[self.rsc_inter] if isinstance(self.rsc_inter,str) else self.rsc_inter)
        self.sb_rsc_p.setValue(self.rsc_p)
        # RandomCropNearBBox
        self.chb_rcnbb_select.setChecked(self.rcnbb_select)
        self.cob_rcnbb_mode.setCurrentIndex(self.MODE[self.rcnbb_mode] if isinstance(self.rcnbb_mode,str) else self.rcnbb_mode)
        self.sb_rcnbb_mps0.setValue(self.rcnbb_mps[0])
        self.sb_rcnbb_mps1.setValue(self.rcnbb_mps[1])
        self.sb_rcnbb_p.setValue(self.rcnbb_p)


        # Crop
        self.chb_crop_select.setChecked(self.crop_select)
        self.cob_crop_mode.setCurrentIndex(self.MODE[self.crop_mode] if isinstance(self.crop_mode,str) else self.crop_mode)
        self.sb_crop_xmi.setValue(self.crop_xmi)
        self.sb_crop_ymi.setValue(self.crop_ymi)
        self.sb_crop_xma.setValue(self.crop_xma)
        self.sb_crop_yma.setValue(self.crop_yma)
        self.sb_crop_p.setValue(self.crop_p)
        # CenterCrop
        self.chb_cc_select.setChecked(self.cc_select)
        self.cob_cc_mode.setCurrentIndex(self.MODE[self.cc_mode] if isinstance(self.cc_mode,str) else self.cc_mode)
        self.sb_cc_height.setValue(self.cc_height)
        self.sb_cc_width.setValue(self.cc_width)
        self.sb_cc_p.setValue(self.cc_p)
        # CropAndPad
        self.chb_cap_select.setChecked(self.cap_select)
        self.cob_cap_mode.setCurrentIndex(self.MODE[self.cap_mode] if isinstance(self.cap_mode,str) else self.cap_mode)
        self.sb_cap_px.setValue(self.cap_px)
        self.sb_cap_percent.setValue(self.cap_percent)
        self.cob_cap_pm.setCurrentIndex(self.border[self.cap_pm] if isinstance(self.cap_pm,str) else self.cap_pm)
        self.sb_cap_pc.setValue(self.cap_pc)
        self.chb_cap_ks.setChecked(self.cap_ks)
        self.chb_cap_si.setChecked(self.cap_si)
        self.cob_cap_inter.setCurrentIndex(self.interpolation[self.cap_inter] if isinstance(self.cap_inter,str) else self.cap_inter)
        self.sb_cap_p.setValue(self.cap_p)
        # Flip
        self.chb_flip_select.setChecked(self.flip_select)
        self.cob_flip_mode.setCurrentIndex(self.MODE[self.flip_mode] if isinstance(self.flip_mode,str) else self.flip_mode)
        self.sb_flip_p.setValue(self.flip_p)
        # Affine
        self.chb_affine_select.setChecked(self.affine_select)
        self.cob_affine_mode.setCurrentIndex(self.MODE[self.affine_mode] if isinstance(self.affine_mode,str) else self.affine_mode)
        self.sb_affine_scale0.setValue(self.affine_scale[0])
        self.sb_affine_scale1.setValue(self.affine_scale[1])
        self.sb_affine_tpe0.setValue(self.affine_tpe[0])
        self.sb_affine_tpe1.setValue(self.affine_tpe[1])
        self.sb_affine_tpx0.setValue(self.affine_tpx[0])
        self.sb_affine_tpx1.setValue(self.affine_tpx[1])
        self.sb_affine_rotate0.setValue(self.affine_rotate[0])
        self.sb_affine_rotate1.setValue(self.affine_rotate[1])
        self.sb_affine_shear0.setValue(self.affine_shear[0])
        self.sb_affine_shear1.setValue(self.affine_shear[1])
        self.cob_affine_inter.setCurrentIndex(self.interpolation[self.affine_inter] if isinstance(self.affine_inter,str) else self.affine_inter)
        self.sb_affine_cval.setValue(self.affine_cval)
        self.cob_affine_mod.setCurrentIndex(self.border[self.affine_mod] if isinstance(self.affine_mod,str) else self.affine_mod)
        self.sb_affine_p.setValue(self.affine_p)
        # Resize
        self.chb_resize_select.setChecked(self.resize_select)
        self.cob_resize_mode.setCurrentIndex(self.MODE[self.resize_mode] if isinstance(self.resize_mode,str) else self.resize_mode)
        self.sb_resize_height.setValue(self.resize_height)
        self.sb_resize_width.setValue(self.resize_width)
        self.cob_resize_inter.setCurrentIndex(self.interpolation[self.resize_inter] if isinstance(self.resize_inter,str) else self.resize_inter)
        self.sb_resize_p.setValue(self.resize_p)
        # Rotate
        self.chb_rotate_select.setChecked(self.rotate_select)
        self.cob_rotate_mode.setCurrentIndex(self.MODE[self.rotate_mode] if isinstance(self.rotate_mode,str) else self.rotate_mode)
        self.sb_rotate_limit0.setValue(self.rotate_limit[0])
        self.sb_rotate_limit1.setValue(self.rotate_limit[1])
        self.cob_rotate_inter.setCurrentIndex(self.interpolation[self.rotate_inter] if isinstance(self.rotate_inter,str) else self.rotate_inter)
        self.cob_rotate_bm.setCurrentIndex(self.rotate_bm)

        self.chb_rotate_cb.setChecked(self.rotate_cb)
        self.sb_rotate_p.setValue(self.rotate_p)
        # Transpose
        self.chb_trans_select.setChecked(self.trans_select)
        self.cob_trans_mode.setCurrentIndex(self.MODE[self.trans_mode] if isinstance(self.trans_mode,str) else self.trans_mode)
        self.sb_trans_p.setValue(self.trans_p)
        
        # 数据处理
        ## easydl
        self.line_edit_cookie.setPlainText(self.cookie)
        self.line_edit_dataset_id.setText(self.dataset_id)
        self.line_edit_saved_path.setText(self.saved_path)
        
        ## 数据转换
        self.le_trans_image_dir.setText(self.trans_image_dir)
        self.le_trans_anno_dir.setText(self.trans_anno_dir)
        self.le_trans_label.setText(self.trans_label)
        self.cob_trans_src_format.setCurrentIndex(self.trans_src_format)
        self.cob_trans_dst_format.setCurrentIndex(self.trans_dst_format)
        self.le_trans_output_dir.setText(self.trans_output_dir)

    def getVar(self):
        self.cm_getVar()
        self.ia_getVar()
        self.mi_getVar()
        aug_list = self.pte_aug_method.toPlainText()
        if len(aug_list) > 0:
            self.aug_lists = eval(aug_list)
        else:
            self.aug_lists = []
        self.advancedblur_select = self.checkbox_advancedblur.isChecked()
        # 模式
        self.advancedblur_mode = self.cb_advancedblur_mode.currentText()
        self.advancedblur_blur = [self.sb_advancedblur_blur_l.value(),self.sb_advancedblur_blur_h.value()]
        self.advancedblur_p = self.sb_advancedblur_p.value()
        self.ab_sigmax = [self.sb_ab_sigmax_l.value(),self.sb_ab_sigmax_h.value()]
        self.ab_sigmay = [self.sb_ab_sigmay_l.value(),self.sb_ab_sigmay_h.value()]
        self.ab_beta_limit = [self.sb_ab_beta_limit_l.value(),self.sb_ab_beta_limit_h.value()]
        self.ab_noise_limit = [self.le_ab_noise_limit_l.value(),self.le_ab_noise_limit_h.value()]
        self.ab_rotate_limit = int(self.le_ab_rotate_limit.text())
        # blur
        # class albumentations.augmentations.blur.transforms.Blur (blur_limit=7, always_apply=False, p=0.5)
        self.blur_select = self.cb_blur_select.isChecked()
        # 模式
        self.blur_mode = self.cb_blur_mode.currentText()
        self.blur_blur_limit = [self.sb_blur_blur_limit0.value(),self.sb_blur_blur_limit1.value()]
        self.blur_p = self.sb_blur_p.value()
        # Defocus
        # class albumentations.augmentations.blur.transforms.Defocus (radius=(3, 10), alias_blur=(0.1, 0.5), always_apply=False, p=0.5)
        self.defocus_select = self.chb_defocus_select.isChecked()
        # 模式
        self.defocus_mode = self.cob_defocus_mode.currentText()
        self.defocus_radius = [self.sb_defocus_radius0.value(),self.sb_defocus_radius1.value()]
        self.defocus_alias_blur = [self.sb_defocus_alias_blur0.value(),self.sb_defocus_alias_blur1.value()]
        self.defocus_p = self.sb_defocus_p.value()
        # GaussianBlur
        # class albumentations.augmentations.blur.transforms.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5)
        self.gab_select = self.chb_gab_select.isChecked()
        # 模式
        self.gab_mode = self.cob_gab_mode.currentText()
        self.gab_blur_limit = [self.sb_gab_blur_limit0.value(),self.sb_gab_blur_limit1.value()]
        self.gab_sigma_limit = [self.sb_gab_sigma_limit0.value(),self.sb_gab_sigma_limit1.value()]
        self.gab_p = self.sb_gab_p.value()
        # GlassBlur
        # class albumentations.augmentations.blur.transforms.GlassBlur (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5)
        self.glb_select = self.chb_glb_select.isChecked()
        # 模式
        self.glb_mode = self.cob_glb_mode.currentText()
        self.glb_sigma = self.sb_glb_sigma.value()
        self.glb_max_delta = self.sb_glb_max_delta.value()
        self.glb_iterations = self.sb_glb_iterations.value()
        self.glb_mode1 = self.cob_glb_mode1.currentText()
        self.glb_p = self.sb_glb_p.value()
        # MedianBlur
        # class albumentations.augmentations.blur.transforms.MedianBlur (blur_limit=7, always_apply=False, p=0.5)
        self.meb_select = self.chb_meb_select.isChecked()
        # 模式
        self.meb_mode = self.cob_meb_mode.currentText()
        self.meb_blur_limit = self.sb_meb_blur_limit.value()
        self.meb_p = self.sb_meb_p.value()
        # MotionBlur
        # class albumentations.augmentations.blur.transforms.MotionBlur (blur_limit=7, allow_shifted=True, always_apply=False, p=0.5) 
        self.mob_select = self.chb_mob_select.isChecked()
        # 模式
        self.mob_mode = self.cob_mob_mode.currentText()
        self.mob_blur_limit = self.sb_mob_blur_limit.value()
        self.mob_allow_shifted = self.chb_mob_allow_shifted.isChecked()
        self.mob_p = self.sb_mob_p.value()
        # ZoomBlur
        # class albumentations.augmentations.blur.transforms.ZoomBlur (max_factor=1.31, step_factor=(0.01, 0.03), always_apply=False, p=0.5)
        self.zb_select = self.chb_zb_select.isChecked()
        # 模式
        self.zb_mode = self.cob_zb_mode.currentText()
        self.zb_max_factor = [self.sb_zb_max_factor0.value(),self.sb_zb_max_factor1.value()]
        self.zb_step_factor = [self.sb_zb_step_factor0.value(),self.sb_zb_step_factor1.value()]
        self.zb_p = self.sb_zb_p.value()
        # RandomBrightness
        self.rb_select = self.chb_rb_select.isChecked()
        self.rb_mode = self.cob_rb_mode.currentText()
        self.rb_limit = [self.sb_rb_limit0.value(),self.sb_rb_limit1.value()]
        self.rb_p = self.sb_rb_p.value()
        # RandomBrightnessContrast
        self.rbc_select = self.chb_rbc_select.isChecked()
        self.rbc_mode = self.cob_rbc_mode.currentText()
        self.rbc_bl = [self.sb_rbc_bl0.value(),self.sb_rbc_bl1.value()]
        self.rbc_cl = [self.sb_rbc_cl0.value(),self.sb_rbc_cl1.value()]
        self.rbc_bbm = self.chb_rbc_bbm.isChecked()
        self.rbc_p = self.sb_rbc_p.value()
        # # RandomContrast
        # self.rc_select = self.chb_rc_select.isChecked()
        # self.rc_mode = self.cob_rc_mode.currentText()
        # self.rc_limit = [self.sb_rc_limit0.value(),self.sb_rc_limit1.value()]
        # self.rc_p = self.sb_rc_p.value()
        # RandomFog 
        self.rf_select = self.chb_rf_select.isChecked()
        self.rf_mode = self.cob_rf_mode.currentText()
        self.rf_fcl = self.sb_rf_fcl.value()
        self.rf_fcu = self.sb_rf_fcu.value()
        self.rf_ac = self.sb_rf_ac.value()
        self.rf_p = self.sb_rf_p.value()
        # RandomGamma
        self.rga_select = self.chb_rga_select.isChecked()
        self.rga_mode = self.cob_rga_mode.currentText()
        self.rga_gl = [self.sb_rga_gl0.value(),self.sb_rga_gl1.value()]
        self.rga_p = self.sb_rga_p.value()
        # # RandomGravel
        # self.rgr_select = self.chb_rgr_select.isChecked()
        # self.rgr_mode = self.cob_rgr_mode.currentText()
        # self.rgr_gr = [self.sb_rgr_gr0.value(),self.sb_rgr_gr1.value(),self.sb_rgr_gr2.value(),self.sb_rgr_gr3.value()]
        # self.rgr_nop = self.sb_rgr_nop.value()
        # self.rgr_p = self.sb_rgr_p.value()
        # # RandomGridShuffle
        # self.rgs_select = self.chb_rgs_select.isChecked()
        # self.rgs_mode = self.cob_rgs_mode.currentText()
        # self.rgs_grid = [self.sb_rgs_grid0.value(),self.sb_rgs_grid1.value()]
        # self.rgs_p = self.sb_rgs_p.value()
        # RandomRain
        self.rr_select = self.chb_rr_select.isChecked()
        self.rr_mode = self.cob_rr_mode.currentText()
        self.rr_sl = self.sb_rr_sl.value()
        self.rr_su = self.sb_rr_su.value()
        self.rr_dl = self.sb_rr_dl.value()
        self.rr_dw = self.sb_rr_dw.value()
        self.rr_dc = [self.sb_rr_dc0.value(),self.sb_rr_dc1.value()]
        self.rr_bv = self.sb_rr_bv.value()
        self.rr_bc = self.sb_rr_bc.value()
        self.rr_p = self.sb_rr_p.value()
        
        # RandomShadow
        self.rsh_select = self.chb_rsh_select.isChecked()
        self.rsh_mode = self.cob_rsh_mode.currentText()
        self.rsh_sr = [self.sb_rsh_sr0.value(),self.sb_rsh_sr1.value(),self.sb_rsh_sr2.value(),self.sb_rsh_sr3.value()]
        self.rsh_nsl = self.sb_rsh_nsl.value()
        self.rsh_nsu = self.sb_rsh_nsu.value()
        self.rsh_sd = self.sb_rsh_sd.value()
        self.rsh_p = self.sb_rsh_p.value()
        # RandomSnow
        self.rsn_select = self.chb_rsn_select.isChecked()
        self.rsn_mode = self.cob_rsn_mode.currentText()
        self.rsn_spl = self.sb_rsn_spl.value()
        self.rsn_spu = self.sb_rsn_spu.value()
        self.rsn_bc = self.sb_rsn_bc.value()
        self.rsn_p = self.sb_rsn_p.value()
        # RandomSunFlare
        self.rsf_select = self.chb_rsf_select.isChecked()
        self.rsf_mode = self.cob_rsf_mode.currentText()
        self.rsf_fr = [self.sb_rsf_fr0.value(),self.sb_rsf_fr1.value(),self.sb_rsf_fr2.value(),self.sb_rsf_fr3.value()]
        self.rsf_al = self.sb_rsf_al.value()
        self.rsf_au = self.sb_rsf_au.value()
        self.rsf_nfcl = self.sb_rsf_nfcl.value()
        self.rsf_sc = [self.sb_rsf_sc0.value(),self.sb_rsf_sc1.value(),self.sb_rsf_sc2.value()]
        self.rsf_nfcu = self.sb_rsf_nfcu.value()
        self.rsf_sr = self.sb_rsf_sr.value()
        self.rsf_p = self.sb_rsf_p.value()
        # RandomToneCurve
        self.rtc_select = self.chb_rtc_select.isChecked()
        self.rtc_mode = self.cob_rtc_mode.currentText()
        self.rtc_s = self.sb_rtc_s.value()
        self.rtc_p = self.sb_rtc_p.value()
        # RandomCrop
        self.rcr_select = self.chb_rcr_select.isChecked()
        self.rcr_mode = self.cob_rcr_mode.currentText()
        self.rcr_height = self.sb_rcr_height.value()
        self.rcr_width = self.sb_rcr_width.value()
        self.rcr_p = self.sb_rcr_p.value()
        # RandomRotate90
        self.rr9_select = self.chb_rr9_select.isChecked()
        self.rr9_mode = self.cob_rr9_mode.currentText()
        self.rr9_p = self.sb_rr9_p.value()
        # RandomResizedCrop
        self.rrc_select = self.chb_rrc_select.isChecked()
        self.rrc_mode = self.cob_rrc_mode.currentText()
        self.rrc_height = self.sb_rrc_height.value()
        self.rrc_width = self.sb_rrc_width.value()
        self.rrc_scale = [self.sb_rrc_scale0.value(),  self.sb_rrc_scale1.value()]
        self.rrc_ratio = [self.sb_rrc_ratio0.value(),  self.sb_rrc_ratio1.value()]
        self.rrc_inter = self.interpolation[self.cob_rrc_inter.currentText()]
        self.rrc_p = self.sb_rrc_p.value()
        # BBoxSafeRandomCrop
        self.bbsrc_select = self.chb_bbsrc_select.isChecked()
        self.bbsrc_mode = self.cob_bbsrc_mode.currentText()
        self.bbsrc_er = self.sb_bbsrc_er.value()
        self.bbsrc_p = self.sb_bbsrc_p.value()
        # RandomCropFromBorders
        self.rcfb_select = self.chb_rcfb_select.isChecked()
        self.rcfb_mode = self.cob_rcfb_mode.currentText()
        self.rcfb_cl = self.sb_rcfb_cl.value()
        self.rcfb_cr = self.sb_rcfb_cr.value()
        self.rcfb_ct = self.sb_rcfb_ct.value()
        self.rcfb_cb = self.sb_rcfb_cb.value()
        self.rcfb_p = self.sb_rcfb_p.value()
        # RandomSizedBBoxSafeCrop	
        self.rsbbsc_select = self.chb_rsbbsc_select.isChecked()
        self.rsbbsc_mode = self.cob_rsbbsc_mode.currentText()
        self.rsbbsc_height = self.sb_rsbbsc_height.value()
        self.rsbbsc_width = self.sb_rsbbsc_width.value()
        self.rsbbsc_er = self.sb_rsbbsc_er.value()
        self.rsbbsc_inter = self.interpolation[self.cob_rsbbsc_inter.currentText()]
        self.rsbbsc_p = self.sb_rsbbsc_p.value()
        # RandomScale
        self.rs_select = self.chb_rs_select.isChecked()
        self.rs_mode = self.cob_rs_mode.currentText()
        self.rs_sl = [self.sb_rs_sl0.value(), self.sb_rs_sl1.value()]
        self.rs_inter = self.interpolation[self.cob_rs_inter.currentText()]
        self.rs_p = self.sb_rs_p.value()
        # RandomSizedCrop
        self.rsc_select = self.chb_rsc_select.isChecked()
        self.rsc_mode = self.cob_rsc_mode.currentText()
        self.rsc_mmh = [self.sb_rsc_mmh0.value(),  self.sb_rsc_mmh1.value()]
        self.rsc_height = self.sb_rsc_height.value()
        self.rsc_width = self.sb_rsc_width.value()
        self.rsc_wr = self.sb_rsc_wr.value()
        self.rsc_inter = self.interpolation[self.cob_rsc_inter.currentText()]
        self.rsc_p = self.sb_rsc_p.value()
        # RandomCropNearBBox
        self.rcnbb_select = self.chb_rcnbb_select.isChecked()
        self.rcnbb_mode = self.cob_rcnbb_mode.currentText()
        self.rcnbb_mps = [self.sb_rcnbb_mps0.value(),  self.sb_rcnbb_mps1.value()]
        self.rcnbb_p = self.sb_rcnbb_p.value()
        # Crop
        self.crop_select = self.chb_crop_select.isChecked()
        self.crop_mode = self.cob_crop_mode.currentText()
        self.crop_xmi = self.sb_crop_xmi.value()
        self.crop_ymi = self.sb_crop_ymi.value()
        self.crop_xma = self.sb_crop_xma.value()
        self.crop_yma = self.sb_crop_yma.value()
        self.crop_p = self.sb_crop_p.value()
        # CenterCrop
        self.cc_select = self.chb_cc_select.isChecked()
        self.cc_mode = self.cob_cc_mode.currentText()
        self.cc_height = self.sb_cc_height.value()
        self.cc_width = self.sb_cc_width.value()
        self.cc_p = self.sb_cc_p.value()
        # CropAndPad
        self.cap_select = self.chb_cap_select.isChecked()
        self.cap_mode = self.cob_cap_mode.currentText()
        self.cap_px = self.sb_cap_px.value()
        self.cap_percent = self.sb_cap_percent.value()
        self.cap_pm = self.border[self.cob_cap_pm.currentText()]
        self.cap_pc = self.sb_cap_pc.value()
        self.cap_ks = self.chb_cap_ks.isChecked()
        self.cap_si = self.chb_cap_si.isChecked()
        self.cap_inter = self.interpolation[self.cob_cap_inter.currentText()]
        self.cap_p = self.sb_cap_p.value()
        # Flip
        self.flip_select = self.chb_flip_select.isChecked()
        self.flip_mode = self.cob_flip_mode.currentText()
        self.flip_p = self.sb_flip_p.value()
        # Affine
        self.affine_select = self.chb_affine_select.isChecked()
        self.affine_mode = self.cob_affine_mode.currentText()
        self.affine_scale = [self.sb_affine_scale0.value(),  self.sb_affine_scale1.value()]
        self.affine_tpe = [self.sb_affine_tpe0.value(),  self.sb_affine_tpe1.value()]
        self.affine_tpx = [self.sb_affine_tpx0.value(),  self.sb_affine_tpx1.value()]
        self.affine_rotate = [self.sb_affine_rotate0.value(),  self.sb_affine_rotate1.value()]
        self.affine_shear = [self.sb_affine_shear0.value(),  self.sb_affine_shear1.value()]
        self.affine_inter = self.interpolation[self.cob_affine_inter.currentText()]
        self.affine_cval = self.sb_affine_cval.value()
        self.affine_mod = self.border[self.cob_affine_mod.currentText()]
        self.affine_p = self.sb_affine_p.value()
        # scale = self.affine_scale, translate_percent = self.affine_tpe,translate_px = self.affine_tpx,rotate = self.affine_rotate,shear = self.affine_shear
        # Resize
        self.resize_select = self.chb_resize_select.isChecked()
        self.resize_mode = self.cob_resize_mode.currentText()
        self.resize_height = self.sb_resize_height.value()
        self.resize_width = self.sb_resize_width.value()
        self.resize_inter = self.interpolation[self.cob_resize_inter.currentText()]
        self.resize_p = self.sb_resize_p.value()
        # Rotate
        self.rotate_select = self.chb_rotate_select.isChecked()
        self.rotate_mode = self.cob_rotate_mode.currentText()
        self.rotate_limit = [self.sb_rotate_limit0.value(),  self.sb_rotate_limit1.value()]
        self.rotate_inter = self.interpolation[self.cob_rotate_inter.currentText()]
        self.rotate_bm = self.border[self.cob_rotate_bm.currentText()]
        self.rotate_cb = self.chb_rotate_cb.isChecked()
        self.rotate_p = self.sb_rotate_p.value()
        # Transpose
        self.trans_select = self.chb_trans_select.isChecked()
        self.trans_mode = self.cob_trans_mode.currentText()
        self.trans_p = self.sb_trans_p.value()

    def show_src_image(self):
        if not os.path.isfile(self.image_path): # and self.image_path.lower().endswiths(["jpg","jpeg","png","bmp"]):
            return 
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.dataset_format = self.cob_dataset_format.currentText()
        if os.path.splitext(self.anno_path)[1].lower() == self.DATA_FORMAT[self.dataset_format]:
            labels,bbox = self.divid_into_label_and_bbox(self.anno_path)
            for i in range(len(bbox)):
                lt,rb=self.tobbox(bbox[i],format = self.dataset_format,size = image.shape[::-1])
                cv2.rectangle(image,lt,rb,color=[0,0,255],thickness=2)
        
        x = image.shape[1]  # 获取图像大小
        y = image.shape[0]
        self.l_src_wh.setText(f"w:{x}, h:{y}")
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(image, x, y, x * 3, QImage.Format_RGB888)
        
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        
        self.scene = QGraphicsScene(self)
        self.scene.addItem(item)
        self.view_src_image.setScene(self.scene)
        self.view_src_image.fitInView(item)    #图像自适应大小
        

    def initUI(self):
        self.cm_detectVar()
        self.ia_detectVar()
        self.mi_detectVar()
        self.button_image_path.clicked.connect(self.open_image_file)
        self.button_anno_path.clicked.connect(self.open_anno_file)
        self.cob_dataset_format.currentIndexChanged.connect(self.show_src_image)
        
        # batch generate
        self.button_image_folder_path.clicked.connect(self.open_image_folder)
        self.button_anno_folder_path.clicked.connect(self.open_anno_folder)
        self.btn_aug_output_dir.clicked.connect(self.open_aug_output_dir)
        self.btn_batch_generate.clicked.connect(self.batch_generate)
        
        # 数据增强
        # 添加
        self.btn_add_method.clicked.connect(self.add_method)
        # 应用图像
        self.btn_apply_image.clicked.connect(self.apply_image)
        # 清空变换
        self.btn_clear_aug.clicked.connect(self.clear_aug)
        # 切换图像
        self.btn_change_image.clicked.connect(self.apply_image)
        # 清空方式勾选
        self.btn_clear_select.clicked.connect(self.clear_select)
        # 删除增强方式
        self.btn_aug_delete.clicked.connect(self.aug_delete)
        self.btn_import_config.clicked.connect(self.import_config)
        self.btn_export_config.clicked.connect(self.export_config)

        # 修改增强方式
        self.btn_edit_aug.clicked.connect(self.set_auglist)


        # 数据处理
        ## easydl
        self.button_set_cookie_path.clicked.connect(self.open_cookie_file)
        self.button_saved_path.clicked.connect(self.open_saved_file)
        self.line_edit_dataset_id.textChanged.connect(self.set_dataset_id)
        self.line_edit_cookie.textChanged.connect(self.set_cookie)
        self.button_download_dataset.clicked.connect(self.download_dataset)
        
        # 格式转换
        self.btn_trans_image_dir_select.clicked.connect(
            lambda: self.le_trans_image_dir.setText(QFileDialog.getExistingDirectory(self,"选取图像文件夹",os.getcwd())))
        self.btn_trans_anno_dir_select.clicked.connect(
            lambda: self.le_trans_anno_dir.setText(QFileDialog.getExistingDirectory(self,"选取标注文件夹",os.getcwd())))
        self.btn_trans_label_select.clicked.connect(
            lambda: self.le_trans_label.setText(QFileDialog.getOpenFileName(self,"选取类别文件",os.getcwd())[0]))
        self.le_trans_label.textChanged.connect(self.set_trans_label)
        self.btn_trans_output_dir_select.clicked.connect(
            lambda: self.le_trans_output_dir.setText(QFileDialog.getExistingDirectory(self,"选取输出文件夹",os.getcwd())))
        self.btn_trans.clicked.connect(self.format_trans)
            
    def updateUI(self):
        # 图片文件路径
        if self.image_path:
            self.line_edit_image_path.setText(self.image_path)
        # 图片文件夹路径
        if self.image_folder_path:
            self.line_edit_image_folder_path.setText(self.image_folder_path)
        # 标注文件路径
        if self.anno_path:
            self.line_edit_anno_path.setText(self.anno_path)
        # 标注文件夹路径
        if self.anno_folder_path:
            self.line_edit_anno_folder_path.setText(self.anno_folder_path)
        
        if self.cookie:
            self.line_edit_cookie.setText(self.cookie)
        if self.saved_path:
            self.line_edit_saved_path.setText(self.saved_path)

    # 修改增强方式参数，并显示到界面
    def set_auglist(self):
        text = self.pte_aug_method.toPlainText()
        try:
            self.aug_lists = eval(text)
            self.apply_image()
        except Exception as e:
            print(f"设置出错，{e}")
        
        
    def open_image_file(self):
        self.image_path,_ = QFileDialog.getOpenFileName(self,"选取图片文件",os.getcwd())
        self.updateUI()
        self.show_src_image()
    def open_anno_file(self):
        self.anno_path,_ = QFileDialog.getOpenFileName(self,"选取标注文件",os.getcwd())
        self.updateUI()
        self.show_src_image()
        
    def open_cookie_file(self):
        self.cookie,_ = QFileDialog.getOpenFileName(self,"选取Cookie文件",os.path.join(os.getcwd(),"config","cookie.txt"))
        self.updateUI()

    def open_image_folder(self):
        self.image_folder_path = QFileDialog.getExistingDirectory(self,"选取图像文件夹",os.getcwd())
        self.updateUI()
    def open_anno_folder(self):
        self.anno_folder_path = QFileDialog.getExistingDirectory(self,"选取标注文件夹",os.getcwd())
        if os.path.isfile(self.anno_folder_path):
            self.line_edit_anno_folder_path.setText(self.anno_folder_path)
        self.updateUI()

    def open_saved_file(self):
        self.saved_path = QFileDialog.getExistingDirectory(self,"选取保存文件夹",os.getcwd())
        self.updateUI()
    def open_aug_output_dir(self):
        self.aug_output_dir = QFileDialog.getExistingDirectory(self,"选取保存文件夹",os.getcwd())
        if os.path.isdir(self.aug_output_dir):
            self.le_aug_output_dir.setText(self.aug_output_dir)

    @staticmethod
    def divid_into_label_and_bbox(file_path):
        labels = []
        bboxes = []

        # 读取标注文件并解析
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                label = int(line[0])
                x_center = float(line[1])
                y_center = float(line[2])
                width = float(line[3])
                height = float(line[4])

                # 将标签和BBox信息存入对应的列表
                labels.append(label)
                bboxes.append([x_center, y_center, width, height])
            return labels,bboxes
    @staticmethod  
    def save_to_txt(labels, bboxes, file_path):
        with open(file_path, 'w') as f:
            for i in range(len(labels)):
                line = f"{labels[i]} {bboxes[i][0]} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]}\n"
                f.write(line)

    @staticmethod
    def tobbox(bbox,format="yolo",size=None):
        if format =="yolo":
            # x_c,y_c,w,h -> xmin ymin,xmax,ymax
            assert size is not None, print("需要输入图片确定图片大小")
            image_w,image_h = size[1:]
            w = bbox[2]*image_w
            h = bbox[3]*image_h
            xmin = bbox[0]*image_w - w/2
            ymin = bbox[1]*image_h - h/2
            xmax = bbox[0]*image_w + w/2
            ymax = bbox[1]*image_h + h/2
            return (int(xmin),int(ymin)),(int(xmax),int(ymax))
        if format == "pascal_voc":
            return (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3]))
        if format == "coco":
            return (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
        if format == "albumentations":
            assert size is not None, print("需要输入图片确定图片大小")
            image_w,image_h = size[1:]
            return (int(bbox[0]*image_w),int(bbox[1])*image_h),(int(bbox[2]*image_w),int(bbox[3])*image_h)
            
            

    def batch_generate(self):
        supported_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        def is_image(file_path):
            _, extension = os.path.splitext(file_path)
            return extension.lower() in supported_image_extensions
        self.pte_batch_log.setPlainText("开始批量增强...")
        imagelists = os.listdir(self.image_folder_path)
        annolists = os.listdir(self.anno_folder_path)
        for img in imagelists:
            
            if not is_image(img):
                continue
            filename, extension = os.path.splitext(img)
            now = datetime.datetime.now()
            # 格式化日期和时间字符串
            date = now.strftime("%Y%m%d")
            time = now.strftime("%H%M%S")
            fm = filename+"aug_"+date+time
            anno = filename+".txt"
            if anno in annolists:
                img = os.path.join(self.image_folder_path,img)
                anno = os.path.join(self.anno_folder_path,anno)
                image = cv2.imread(img)                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                anno_label,anno_bbox = self.divid_into_label_and_bbox(anno)
                self.images_format = self.cob_datasets_format.currentText()
                result = Compose(self.aug_lists,bbox_params=BboxParams(format=self.images_format, min_area=1024, min_visibility=0.1, label_fields=['class_labels']))(image=image,bboxes=anno_bbox,class_labels=anno_label)
                aug_image =result["image"]
                aug_bboxes = result["bboxes"]
                aug_class_labels = result["class_labels"]
                
                image_dir = os.path.join(self.aug_output_dir,"image")
                anno_dir = os.path.join(self.aug_output_dir,"label")
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                if not os.path.exists(anno_dir):
                    os.makedirs(anno_dir)
                self.save_to_txt(aug_class_labels,aug_bboxes,os.path.join(anno_dir,fm+".txt"))
                dst_image = cv2.cvtColor(aug_image,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_dir,fm+".png"),dst_image)
        self.pte_batch_log.setPlainText("批量增强完成！")
                
                
                
                
    
    # 格式转换
    def format_trans(self):
        src_format = self.cob_trans_src_format.currentText()
        dst_format = self.cob_trans_dst_format.currentText()
        
        image_dir = self.le_trans_image_dir.text()
        anno_dir = self.le_trans_anno_dir.text()
        ouput_dir = self.le_trans_output_dir.text()
        label = self.le_trans_label.text()
        if os.path.isfile(label):
            with open(label, 'r') as f:
                label = [i.strip() for i in f.readlines() if i.strip()!=""]
        else:
            label = [i for i in re.split(r"[\s,;]+",label) if i.strip()!=""]

        lt = LabelTran(src_format,dst_format,label,out_dir=ouput_dir)
        lt(label_folder=anno_dir,image_folder=image_dir if image_dir!="" else None)
        

    # 数据增强
    # 添加按钮处理函数
    def add_method(self): 
        self.getVar()
        oneoflist = []
        compose = []  
            
        if self.advancedblur_select:
        # 模式
            func = AdvancedBlur(blur_limit=self.advancedblur_blur, sigmaX_limit=self.ab_sigmax, sigmaY_limit=self.ab_sigmay, rotate_limit=self.ab_rotate_limit, beta_limit=self.ab_beta_limit, noise_limit=self.ab_noise_limit, p=self.advancedblur_p)
            if self.advancedblur_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # blur
        # class albumentations.augmentations.blur.transforms.Blur (blur_limit=7, always_apply=False, p=0.5)
        if self.blur_select:
        # 模式
            func = Blur(blur_limit=self.blur_blur_limit, p=self.blur_p)
            if self.blur_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Defocus
        # class albumentations.augmentations.blur.transforms.Defocus (radius=(3, 10), alias_blur=(0.1, 0.5), always_apply=False, p=0.5)
        if self.defocus_select:
        # 模式
            func = Defocus(radius=self.defocus_radius,alias_blur=self.defocus_alias_blur,p=self.defocus_p)
            if self.defocus_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # GaussianBlur
        # class albumentations.augmentations.blur.transforms.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5)
        if self.gab_select:
        # 模式
            func = GaussianBlur(blur_limit=self.gab_blur_limit,sigma_limit=self.gab_sigma_limit,p=self.gab_p)
            if self.gab_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # GlassBlur
        # class albumentations.augmentations.blur.transforms.GlassBlur (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5)
        if self.glb_select:
        # 模式
            func = GlassBlur(sigma=self.glb_sigma,max_delta=self.glb_max_delta,iterations=self.glb_iterations, mode =self.glb_mode1, p=self.gab_p)
            if self.glb_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # MedianBlur
        # class albumentations.augmentations.blur.transforms.MedianBlur (blur_limit=7, always_apply=False, p=0.5)
        if self.meb_select:
        # 模式
            func = MedianBlur(blur_limit=self.meb_blur_limit, p=self.meb_p)
            if self.meb_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # MotionBlur
        # class albumentations.augmentations.blur.transforms.MotionBlur (blur_limit=7, allow_shifted=True, always_apply=False, p=0.5) 
        if self.mob_select:
        # 模式
            func = MotionBlur(blur_limit=self.mob_blur_limit,allow_shifted=self.mob_allow_shifted, p=self.mob_p)
            if self.mob_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # ZoomBlur
        # class albumentations.augmentations.blur.transforms.ZoomBlur (max_factor=1.31, step_factor=(0.01, 0.03), always_apply=False, p=0.5)
        if self.zb_select:
        # 模式
            func = ZoomBlur(max_factor=self.zb_max_factor,step_factor=self.zb_step_factor, p=self.zb_p)
            if self.zb_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomBrightness
        if self.rb_select:
            func = RandomBrightness(limit=self.rb_limit,p=self.rb_p)
            if self.rb_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomBrightnessContrast
        if self.rbc_select:
            func = RandomBrightnessContrast(brightness_limit=self.rbc_bl,contrast_limit=self.rbc_cl,brightness_by_max=self.rbc_bbm,p=self.rbc_p)
            if self.rbc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # # RandomContrast
        # if self.rc_select:
        #     func = RandomContrast(limit=self.rc_limit,p=self.rc_p)
        #     if self.rc_mode=="Oneof":
        #         oneoflist.append(func)
        #     else:
        #         compose.append(func)
        # RandomFog 
        if self.rf_select:
            func = RandomFog(fog_coef_lower=self.rf_fcl,fog_coef_upper=self.rf_fcu,alpha_coef=self.rf_ac,p=self.rf_p)
            if self.rf_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomGamma
        if self.rga_select:
            func = RandomGamma(gamma_limit=self.rga_gl,p=self.rga_p)
            if self.rga_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # # RandomGravel
        # if self.rgr_select:
        #     func = A.RandomGravel(gravel_roi=self.rgr_gr,number_of_patches=self.rgr_nop ,p=self.rgr_p)
        #     if self.rgr_mode=="Oneof":
        #         oneoflist.append(func)
        #     else:
        #         compose.append(func)
        # # RandomGridShuffle
        # if self.rgs_select:
        #     func = RandomGridShuffle(grid=self.rgs_grid,p=self.rgs_p)
        #     if self.rgs_mode=="Oneof":
        #         oneoflist.append(func)
        #     else:
        #         compose.append(func)
        # RandomRain
        if self.rr_select:
            func = RandomRain(slant_lower = self.rr_sl, slant_upper = self.rr_su,drop_length = self.rr_dl,drop_width = self.rr_dw,drop_color = self.rr_dc,blur_value = self.rr_bv,brightness_coefficient = self.rr_bc,p = self.rr_p)
            if self.rr_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomShadow
        if self.rsh_select:
            func = RandomShadow(shadow_roi=self.rsh_sr,num_shadows_lower=self.rsh_nsl,num_shadows_upper=self.rsh_nsu,shadow_dimension = self.rsh_sd,p=self.rsh_p)
            if self.rsh_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomSnow
        if self.rsn_select:
            func = RandomSnow(snow_point_lower=self.rsn_spl,snow_point_upper=self.rsn_spu,brightness_coeff=self.rsn_bc,p=self.rsn_p)
            if self.rsn_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomSunFlare
        if self.rsf_select:
            func = RandomSunFlare(flare_roi=self.rsf_fr,angle_lower=self.rsf_al,angle_upper=self.rsf_au,num_flare_circles_lower=self.rsf_nfcl,num_flare_circles_upper=self.rsf_nfcu,src_radius=self.rsf_sr,src_color=self.rsf_sc,p=self.rsf_p)
            if self.rsf_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomToneCurve
        if self.rtc_select:
            func = RandomToneCurve(scale=self.rtc_s,p=self.rtc_p)
            if self.rtc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomCrop
        if self.rcr_select:
            func = RandomCrop(height = self.rcr_height, width = self.rcr_width,p = self.rcr_p)
            if self.rcr_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomRotate90
        if self.rr9_select:
            func = RandomRotate90(p = self.rr9_p)
            if self.rr9_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomResizedCrop
        if self.rrc_select:
            func = RandomResizedCrop(height = self.rrc_height, width = self.rrc_width,scale = self.rrc_scale,ratio = self.rrc_ratio,interpolation = self.rrc_inter,p = self.rrc_p)
            if self.rrc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # BBoxSafeRandomCrop
        if self.bbsrc_select:
            func = BBoxSafeRandomCrop(erosion_rate = self.bbsrc_er, p = self.bbsrc_p)
            if self.bbsrc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomCropFromBorders
        if self.rcfb_select:
            func = RandomCropFromBorders(crop_left = self.rcfb_cl, crop_right = self.rcfb_cr,crop_top = self.rcfb_ct,crop_bottom = self.rcfb_cb,p = self.rcfb_p)
            if self.rcfb_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomSizedBBoxSafeCrop	
        if self.rsbbsc_select:
            func = RandomSizedBBoxSafeCrop(height = self.rsbbsc_height, width = self.rsbbsc_width,erosion_rate = self.rsbbsc_er,interpolation = self.rsbbsc_inter,p = self.rsbbsc_p)
            if self.rsbbsc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomScale
        if self.rs_select:
            func = RandomScale(scale_limit = self.rs_sl, interpolation = self.rs_inter,p = self.rs_p)
            if self.rs_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomSizedCrop
        if self.rsc_select:
            func = RandomSizedCrop(min_max_height = self.rsc_mmh, height = self.rsc_height,width = self.rsc_width,w2h_ratio = self.rsc_wr,interpolation = self.rsc_inter,p = self.rsc_p)
            if self.rsc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # RandomCropNearBBox
        if self.rcnbb_select:
            func = RandomCropNearBBox(max_part_shift = self.rcnbb_mps,p = self.rcnbb_p)
            if self.rcnbb_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Crop
        if self.crop_select:
            func = Crop(x_min = self.crop_xmi, y_min = self.crop_ymi,x_max = self.crop_xma,y_max = self.crop_yma,p = self.crop_p)
            if self.crop_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # CenterCrop
        if self.cc_select:
            func = CenterCrop(height = self.cc_height, width = self.cc_width, p = self.rr_p)
            if self.cc_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # CropAndPad
        if self.cap_select:
            px = self.cap_px
            percent = self.cap_percent
            if self.cap_px == 0:
                px = None
            if self.cap_percent == 0.0:
                percent = None
            if px is None and percent is None:
                px = 30
            func = CropAndPad(px=px, percent=percent, pad_mode = self.cap_pm,pad_cval = self.cap_pc, keep_size = self.cap_ks,sample_independently = self.cap_si,interpolation = self.cap_inter,p = self.cap_p)
            if self.cap_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Flip
        if self.flip_select:
            func = Flip(p = self.flip_p)
            if self.flip_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Affine
        if self.affine_select:
        # scale = self.affine_scale, translate_percent = self.affine_tpe,translate_px = self.affine_tpx,rotate = self.affine_rotate,shear = self.affine_shear
            func = Affine(scale = None, translate_percent = None,translate_px = None, rotate = None,shear = None, interpolation = self.affine_inter,cval = self.affine_cval, mode = self.affine_mod, p = self.affine_p)
            if self.affine_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Resize
        if self.resize_select:
            func = Resize(height = self.resize_height, width = self.resize_width, interpolation = self.resize_inter, p = self.resize_p)
            if self.resize_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Rotate
        if self.rotate_select:
            func = Rotate(limit = self.rotate_limit, interpolation = self.rotate_inter,border_mode = self.rotate_bm,crop_border = self.rotate_cb, p = self.rotate_p)
            if self.rotate_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
        # Transpose
        if self.trans_select:
            func = Transpose(p = self.trans_p)
            if self.trans_mode=="Oneof":
                oneoflist.append(func)
            else:
                compose.append(func)
            
        # 一层一层添加
        if len(oneoflist):
            compose.append(OneOf(oneoflist, p=0.5))
        
        self.aug_lists+=compose
        self.pte_aug_method.setPlainText("["+",\n".join([str(i) for i in self.aug_lists])+"]")
        
        # self.clear_select()
    
    # 数据增强方式应用到图像
    def apply_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                        
        self.image_format = self.cob_dataset_format.currentText()
        self.dataset_format = self.cob_dataset_format.currentText()
        if os.path.splitext(self.anno_path)[1].lower() == self.DATA_FORMAT[self.dataset_format]:
            labels,bbox = self.divid_into_label_and_bbox(self.anno_path)
        else:
            bbox = None
            labels = None
                
        result = Compose(self.aug_lists,bbox_params=BboxParams(format=self.image_format, min_area=1024, min_visibility=0.1, label_fields=['class_labels']))(image=image,bboxes=bbox,class_labels=labels)
        aug_image =result["image"]
        
        if bbox is not None:
            aug_bboxes = result["bboxes"]
            aug_class_labels = result["class_labels"]
            for i in range(len(aug_bboxes)):
                lt,rb=self.tobbox(aug_bboxes[i],format = self.dataset_format,size = aug_image.shape[::-1])
                cv2.rectangle(aug_image,lt,rb,color=[0,0,255],thickness=2)
        
        # for i in range(len(aug_bboxes)):
        #     x1,y1,x2,y2=aug_bboxes[i]
        #     cv2.rectangle(aug_image,(int(x1),int(y1)),(int(x2),int(y2)),color=[0,0,255],thickness=2)
        
        x = aug_image.shape[1]  # 获取图像大小
        y = aug_image.shape[0]
        self.l_dst_wh.setText(f"w:{x}, h:{y}")
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(aug_image, x, y, x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)

        item = QGraphicsPixmapItem(pix)
        
        self.scene = QGraphicsScene(self)
        self.scene.addItem(item)
        
        self.view_transform_image.setScene(self.scene)
        self.view_transform_image.fitInView(item)    #图像自适应大小
        
    def clear_aug(self):
        self.aug_lists = []
        self.pte_aug_method.setPlainText("")        
        
    # 清除勾选
    def clear_select(self):
        self.checkbox_advancedblur.setChecked(0)
        self.cb_blur_select.setChecked(0)
        self.chb_defocus_select.setChecked(0)
        self.chb_gab_select.setChecked(0)
        self.chb_glb_select.setChecked(0)
        self.chb_meb_select.setChecked(0)
        self.chb_mob_select.setChecked(0)
        self.chb_mob_allow_shifted.setChecked(0)
        self.chb_zb_select.setChecked(0)
        self.chb_rb_select.setChecked(0)
        self.chb_rbc_select.setChecked(0)
        self.chb_rbc_bbm.setChecked(0)
        self.chb_rc_select.setChecked(0)
        self.chb_rf_select.setChecked(0)
        self.chb_rga_select.setChecked(0)
        self.chb_rgr_select.setChecked(0)
        self.chb_rgs_select.setChecked(0)
        self.chb_rr_select.setChecked(0)
        self.chb_rsh_select.setChecked(0)
        self.chb_rsn_select.setChecked(0)
        self.chb_rsf_select.setChecked(0)
        self.chb_rtc_select.setChecked(0)
        
        self.chb_rc_select.setChecked(0)
        self.chb_rr9_select.setChecked(0)
        self.chb_rrc_select.setChecked(0)
        self.chb_bbsrc_select.setChecked(0)
        self.chb_rcfb_select.setChecked(0)
        self.chb_rsbbsc_select.setChecked(0)
        self.chb_rs_select.setChecked(0)
        self.chb_rsc_select.setChecked(0)
        self.chb_rcnbb_select.setChecked(0)
        self.chb_crop_select.setChecked(0)
        self.chb_cc_select.setChecked(0)
        self.chb_cap_select.setChecked(0)
        self.chb_flip_select.setChecked(0)
        self.chb_affine_select.setChecked(0)
        self.chb_resize_select.setChecked(0)
        self.chb_rotate_select.setChecked(0)
        self.chb_trans_select.setChecked(0)

    def aug_delete(self):
        index = self.sb_aug_index.value()
        if index < len(self.aug_lists):
            del self.aug_lists[index]
        self.pte_aug_method.setPlainText("["+",\n".join([str(i) for i in self.aug_lists])+"]")
        

    # 数据处理
    # easydl
    def set_dataset_id(self):
        self.dataset_id = self.line_edit_dataset_id.text()
    def set_cookie(self):
        self.cookie = self.line_edit_cookie.toPlainText()
    def set_trans_label(self):
        self.trans_label = self.le_trans_label.text()
    def download_dataset(self):
        ed = EasyDL(cookie_file=self.cookie)
        self.text_brower_easydl_log.insertPlainText(f"开始下载：{self.dataset_id}, 保存路径：{self.saved_path}\n")
        ed.downloaddateset(int(self.dataset_id),self.saved_path)
        self.text_brower_easydl_log.insertPlainText(f"数据下载完成\n")



if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        # MainWindow = QMainWindow()
        ui = DataAug()
        # ui = ui_mainwindows.Ui_MainWindow()  
        # ui.setupUi(MainWindow)
        ui.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"发生错误，{e}")

