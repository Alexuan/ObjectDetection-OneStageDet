# -*- coding:utf-8 -*-
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import pickle

from os import listdir, getcwd
from os.path import join
import math

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2019', 'Annotations')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["mn"]

def convert(size, box,image_id):
    try:
        dw = 1./size[0]
        dh = 1./size[1]
    except:
        dw=dh=400
        print(image_id)
    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]

    cx = cx*dw
    w = w*dw
    cy = cy*dh
    h = h*dh
    # 0~0.5pi不变，0.5pi到pi要 -pi
    angle = box[4]
    if angle > 0.5*math.pi:
        angle = angle - math.pi
    # -90到90度进行归一化
    angle = (angle + 0.5*math.pi)/math.pi
    return (cx,cy,w,h,angle)

def parse_rec(xml_path):
    '''读取drink xml的标记文件
    '''
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.findall('object'):
        # print ('elem in obj:===========')
        # for elem in obj:
        #     print elem.tag, elem.attrib
        obj_struct = {}
        # print (obj.find('type').text)
        
        if ('YL' not in obj.find('name').text) and obj.find('name').text != '0':
            print ('YL not in :',obj.find('name').text)
            continue
        type = obj.find('type').text
        if type == 'bndbox':
            obj_struct['type'] = 'bndbox'
            obj_struct['name'] = obj.find('name').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')

            xmin = float(bbox.find('xmin').text)
            xmax = float(bbox.find('xmax').text)
            ymin = float(bbox.find('ymin').text)
            ymax = float(bbox.find('ymax').text)

            xcenter = (xmin + xmax) / 2.0
            ycenter = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin

            obj_struct['rbbox'] = [xcenter,
                                  ycenter,
                                  w,
                                  h,
                                  0]
            objects.append(obj_struct)
            # print(obj_struct)
        elif type == 'robndbox':
            obj_struct['type'] = 'robndbox'
            obj_struct['name'] = obj.find('name').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('robndbox')
            obj_struct['rbbox'] = [float(bbox.find('cx').text),
                                  float(bbox.find('cy').text),
                                  float(bbox.find('w').text),
                                  float(bbox.find('h').text),
                                  float(bbox.find('angle').text)]
            objects.append(obj_struct)
            # print(obj_struct)
        else:
            print('xml bndbox type error')
            continue
    # print (objects)
    return objects

def convert_annotation(data_root,list_file, year, image_set, image_id):
    in_file = open(data_root + '/%s/%s.xml'%(image_set,image_id))

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # print('w=',w, in_file)
    # for elem in tree.iter():

    # for elem in root:
        # print elem.tag, elem.attrib
    # bbox_list = parse_rec(root)       
    bbox_list = parse_rec(data_root + '/%s/%s.xml'%(image_set,image_id))
    out_file = open(data_root + '/labels/%s.txt'%(image_id), 'w')
    if len(bbox_list):
        list_file.write(data_root + '/JPEGImages/%s.jpg\n'%(image_id))
        for obj in bbox_list:
            # difficult = obj.find("attributes").find('difficult0').text
            # cls = obj['name']
            cls = 'mn'
            # if cls not in classes or int(difficult) == 1:
            if cls not in classes:# or int(difficult) == 1
                continue
            cls_id = classes.index(cls)

            b = obj['rbbox']
            bb = convert((w,h), b,image_id)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# wd = getcwd()
data_root = '/data1/condiren/drink/mengniu/allbox/'
for year, image_set in sets:
    if not os.path.exists(data_root + '/labels/'):
        os.makedirs(data_root + '/labels/')
    image_ids = []
    for file_name in os.listdir(data_root + '/%s'%(image_set)):
        file_raw_name = file_name[:file_name.rfind('.')]
        image_ids.append(file_raw_name)

    # image_ids = open(data_root + '/%s.txt'%(image_set)).read().strip().split()
    list_file = open(data_root + '/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        
        convert_annotation(data_root,list_file, year, image_set, image_id)
    list_file.close()

