# VOC转yolo
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from .log_handler import Log
import copy

class LabelTran(object):
    def __init__(self, input='xml',output="txt",label='',out_dir=None) -> None:
        """
        input: 输入的格式
        outpu: 输出的格式
        label: 标签列表或文件
        out_dir: 输出的文件夹位置
        """
        __support_format = ["xml","txt"]
        format_map = {"pascal_voc":"xml","yolo":"txt"}
        log = Log()
        log.info(f"目前仅仅实现voc[xml]转yolo[txt]格式,更多只需要添加解析与编码操作")
        

        self.input=format_map[input]
        self.output=format_map[output]
        assert label is not None, log.warning(f"label is not specified")
        assert self.input in __support_format, log.warning(f"input format{self.input} is not supported{__support_format}")
        assert self.output in __support_format, log.warning(f"output format{self.input} is not supported{__support_format}")
        if isinstance(label,str):
            if os.path.exists(label) and os.path.isfile(label):
                self.label = [i.strip() for i in open(label).readlines()]
                log.info(f"read class name from file[{label}]")
        elif isinstance(label,list):
            self.label = label
            
        log.info(f"{len(self.label)} classes, {', '.join(self.label)}") 
        
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        # 可添加的内容
        # object:label,x,y,w,h
        self.basic_info = {'src_w':0,'src_h':0,'src':'','object':[]}
        # 格式解析的函数，返回上述格式
        self.parse={"xml":'self.XMLParse'}
        # 写入的函数，根据上述格式写入指定格式
        self.encode={"txt":'self.TXTEncode'}
        
    def XMLParse(self,label,image=None):
        """
        parse a xml file
        """
        if self.output=="json":
            info['src'] = ''
            # assert os.path.isfile(label),self.log.warning(f"label[{label}] is not file")
            # 找到图片
            pass
        assert os.path.isfile(label),self.log.warning(f"label[{label}] is not file")
        # info = []
        info = copy.deepcopy(self.basic_info)
        in_file = open(label)
        xml_text = in_file.read()
        root = ET.fromstring(xml_text)
        in_file.close()
        size = root.find('size')
        info['src_w'] = int(size.find('width').text)
        info['src_h'] = int(size.find('height').text)
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in self.label:
                self.log.warning(f"class[{cls}] is not in label[self.label]")
                continue
            cls_id = self.label.index(cls)
            xmlbox = obj.find('bndbox')
            info['object'].append((cls_id,float(xmlbox.find('xmin').text),float(xmlbox.find('ymin').text),float(xmlbox.find('xmax').text)-float(xmlbox.find('xmin').text),float(xmlbox.find('ymax').text)-float(xmlbox.find('ymin').text)))
        return info
    
    def TXTEncode(self,info,filename):
        if self.out_dir is None:
            out_file = open(os.path.splitext(filename)[0]+".txt",'w')
        else:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            out_file = open(os.path.join(self.out_dir,os.path.splitext(os.path.basename(filename))[0]+".txt"),'w')

        def convert(size, box):
            dw = 1.0/size[0]
            dh = 1.0/size[1]
            x = box[0]+box[2]/2.0
            y = box[1]+box[3]/2.0
            w = box[2]
            h = box[3]
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh
            return (x,y,w,h)
        for obj in info['object']:
            bb = convert((info['src_w'],info['src_h']), obj[1:])
            out_file.write(str(obj[0]) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()
                
    
    def __call__(self,label_folder,image_folder=None):
        """
        目前仅支持文件目录
        """
        if self.input == "xml":
            image_folder = None
        xml_lists = [os.path.join(label_folder,i) for i in os.listdir(label_folder) if os.path.splitext(i)[-1] =="."+self.input]
        for i in xml_lists:
            img_path =None
            if self.output=="json":
                for j in [os.path.join(image_folder,i) for i in os.listdir(image_folder) if os.path.splitext(i)[-1] in [".png",".jpg",".jpeg",".bmp"]]:
                    if os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(j))[0]:
                        img_path = j
                        break
            info = eval(self.parse[self.input])(i,img_path)
            eval(self.encode[self.output])(info,i)
 
if __name__ == '__main__':
    lt = LabelTran("xml","txt",['yyzz','yz'],out_dir=r'F:\Project\DA-GUI\output')
    lt(r'F:\Project\DA-GUI\dataset\1754956')