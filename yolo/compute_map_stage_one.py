# -*- coding:utf-8 -*-
# 分析模型输出结果
# 需要每个类别一个结果txt文件，以及测试图像和对应的xml标记文件
# 可以统计map以及输出所有正确检测，误检和漏检的框到对应测试图像上

import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
# from matplotlib import pyplot as plt
import pdb
import glob

VISUAL_FLAG = False
DEBUG = False
dataset = "mn_100"
THRESHOLD = 0.1
user_name = 'zhongyisun'

# 图像输出文件夹
# OUTPUT_IMGS_DIR = "/home/beililiu/data/drink/drink89/result/20180702/bbnd_output"
OUTPUT_IMGS_DIR = "/data/home/%s/data/drink/mn_test/bbnd_output" % user_name
# 检测结果文件夹
# DET_DIR = '/home/beililiu/data/drink/drink89/result/20180702/dets'
#DET_DIR = '/data/home/%s/ab-darknet/results' % user_name
DET_DIR = '/home/caspardu/shixuan/darknet_to_onedet/ObjectDetection-OneStageDet/yolo/results'
# DET_DIR = '/home/beililiu/darknet/results/drink'
# label_map
LABEL_MAP_PATH = '/raid/data1/%s/data/drink/drink_shelf_test/stage_one_label_map.txt' % user_name

# 保存每个图像中检测正确和错误的bbox
classify_right_bboxes = dict()
classify_wrong_bboxes = dict()

# 保存每个图像中检测正确和错误的bbox类别概率
classify_right_score = dict()
classify_wrong_score = dict()

label_map = set()

def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        for line in f.readlines():
            label_map.add(line.strip().split(':')[-1])


def parse_rec(xml_path):
    '''读取drink xml的标记文件
    '''
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # drink
        obj_struct['truncated'] = int(obj.find("attributes").find('truncated').text)
        # obj_struct['difficult'] = int(obj.find("attributes").find('difficult0').text)
        obj_struct['difficult'] = 0
        obj_struct['blocked'] = int(obj.find("attributes").find('blocked').text)

        # dong
        # obj_struct['difficult'] = int(obj.find('difficult').text)
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text),
                              float(bbox.find('ymin').text),
                              float(bbox.find('xmax').text),
                              float(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec):
    '''
    计算ap，采用VOC2010计算方式
    :param rec: recall向量
    :param prec: precision向量
    :return: ap数值
    '''
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    print(mpre)
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    print((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vis_detections(im, img_id, score, cls_name, xmin, ymin, xmax, ymax, color):
    """Draw detected bounding boxes.
    绘制输出框，有特定颜色
    """
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if color == "red":
        image = cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0, 0, 255), 6)
        image = cv2.putText(image,cls_name,(xmax-80,ymax-5),font,1.2,(0, 0, 255),4)
    elif color == "yellow":
        image = cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(40, 150, 255), 6)
        image = cv2.putText(image,cls_name,(xmin,ymin-5),font,1.2,(40, 150, 255),4)
    elif color == "blue":
        image = cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(110, 50, 50), 6)
        image = cv2.putText(image,cls_name,(xmin,ymin-5),font,1.2,(110, 50, 50),4)

    # 由于会对于im进行多次标记
    # 所以此处可能会有多轮保存过程
    img_name = os.path.join(OUTPUT_IMGS_DIR, img_id+'_add_box.jpg')
    cv2.imwrite(img_name, image)


def voc_eval(detpath, ann_dir, classname, img_id_value_map, ovthresh=0.5):
    '''
    根据voc的方式统计模型检测输出结果
    :param detpath: 模型输出结果文件
    :param ann_dir: xml标记文件位置，voc2007/Annotations
    :param classname: 类别名称
    :param img_id_value_map: 测试图像集合
    :param ovthresh: 阈值
    :return: recall和precision的向量，以及ap值
    '''
    imagenames = img_id_value_map.keys()
    img_ann_map_gt = {}
    npos = 0  # 所有gt框的数量
    all_bb_gt = {}
    # 遍历测试list中所有图像，读取标记xml文件
    for imagename in imagenames:
        objs = parse_rec(os.path.join(ann_dir, imagename+".xml"))
        # R = [obj for obj in objs if obj['name'] == classname]
        R = [obj for obj in objs]  # 对于只做单一饮料输出的情况
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        truncated = np.array([x['truncated'] for x in R]).astype(np.bool)
        # blocked = np.array([x['truncated'] for x in R]).astype(np.int8)
        cls_name = [x['name'] for x in R]  # 每个检测框类别名称
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # image id到标记的对应
        img_ann_map_gt[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det,
                                     'class_name': cls_name,
                                     'truncated': truncated}
                                     # 'blocked': blocked}
        for b, d in zip(bbox, difficult):
            if not d:
                b_str = str(b[0])+"_"+str(b[1])+"_"+str(b[2])+"_"+str(b[3])
                all_bb_gt[b_str] = imagename
    # print('gt len:', len(img_ann_map_gt))
    # 读入检测结果文件
    with open(detpath) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        conf = float(line.strip().split(' ')[1])
        if conf >= THRESHOLD:
            new_lines.append(line)
    lines = new_lines
    # 分割处理检测结果文件
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0].replace(".jpg", "") for x in splitlines]  # image id
    confidence = np.array([float(x[1]) for x in splitlines])  # 置信度
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # bbox位置

    nd = len(image_ids)  # 模型输出的检测结果数量
    tp = np.zeros(nd)
    #false positive: incorrectly assign bounding box
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        # 根据score大小降序排序
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        matched_bb_gt = {}

        # 遍历所有模型输出的检测框
        for d in range(nd):
            # if sorted_scores[d] > -0.5:
            #     print(d)
            #     break
            image_id = image_ids[d]  # 检测框所属图像id
            # im = img_id_value_map[image_id]  # 检测框所属图像矩阵
            ann_gt = img_ann_map_gt[image_id]  # 对应图像的gt框
            bb = BB[d, :].astype(float)  # 模型输出检测框
            ovmax = -np.inf
            BBGT = ann_gt['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                # 计算IoU
                overlaps = inters / uni
                # 计算最大IoU以及其对应的gt框编号
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                # IoU大于阈值
                if not ann_gt['difficult'][jmax]:  # difficult框不计入ap计算
                    if not ann_gt['det'][jmax]:
                        # 该gt框没有匹配过，标记其匹配
                        ann_gt['det'][jmax] = 1
                        bb_gt = ann_gt['bbox'][jmax]
                        bb_gt_str = str(bb_gt[0])+"_"+str(bb_gt[1])+"_"+str(bb_gt[2])+"_"+str(bb_gt[3])
                        matched_bb_gt[bb_gt_str] = image_id

                        # 判断分类是否正确
                        # if ann_gt['class_name'][jmax] == classname:
                        if ann_gt['class_name'][jmax]:  # 只对于检测单个饮料类别
                            # 分类正确，记为正样本
                            tp[d] = 1.
                            classify_right_bboxes[image_id].append({'bb': bb,
                                                                    'cls': classname,
                                                                    'score': -sorted_scores[d]})
                        else:
                            # 分类错误，记为负样本
                            fp[d] = 1.
                            classify_wrong_bboxes[image_id].append({'bb': bb,
                                                                    'cls': classname,
                                                                    'score': -sorted_scores[d]})
                            # 记录该检测为类别误检
                    else:
                        # 该gt框已经匹配过，检测结果为重复匹配
                        fp[d] = 1.
                        classify_wrong_bboxes[image_id].append({'bb': bb,
                                                                    'cls': classname,
                                                                    'score': -sorted_scores[d]})

            else:
                # IoU小于阈值，记为负样本
                print(image_id, ovmax)
                fp[d] = 1.
                classify_wrong_bboxes[image_id].append({'bb': bb,
                                                                    'cls': classname,
                                                                    'score': -sorted_scores[d]})

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos+np.finfo(np.float64).eps)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap


def eval(img_dir, ann_dir, imagesetfile):
    '''
    评估图像输出结果
    :param img_dir: 图像目录
    :param ann_dir: xml标记文件目录
    :param imagesetfile:图像列表文件
    :return:
    '''
    aps = []
    recs = []

    img_id_value_map = {}

    # 读入测试列表文件
    with open(imagesetfile) as f:
        lines = f.readlines()
    imagenames = [x.strip().replace(".jpg", "") for x in lines]

    # 读入所有测试图像
    for imagename in tqdm(imagenames):
        classify_right_bboxes[imagename] = []
        classify_wrong_bboxes[imagename] = []
        if VISUAL_FLAG:
            img_id_value_map[imagename] = cv2.imread(img_dir+"/"+imagename+".jpg")
        else:
            img_id_value_map[imagename] = ()

    print(len(img_id_value_map))

    det_dir = DET_DIR

    # 逐个遍历每个类别的标记输出文件，计算该类别的ap和recall
    # 要求模型输出结果按照每个类别组织一个txt文件
    # 从而最后统计map和平均recall
    for detpath in os.listdir(det_dir):
        # 获取标记的名称
        label = detpath.split("/")[-1].replace(".txt", "")
        detpath = os.path.join(det_dir, detpath)
        print(detpath, label)

        rec, prec, ap = voc_eval(detpath, ann_dir, label, img_id_value_map, ovthresh=0.5)
        fmax = 0
        pmax = 0
        rmax = 0
        for r, p in zip(rec, prec):
            f = 2 / (1 / r + 1 / p)
            if f > fmax:
                fmax = f
                pmax = p
                rmax = r
        print('precision: %.4f, recall: %.4f, f1: %.4f' % (pmax, rmax, fmax))
        f = 2.0 / (1.0 / rec[-1] + 1.0 / prec[-1])
        print('precision: %.4f, recall: %.4f, f1: %.4f' % (prec[-1], rec[-1], f))
        # print(prec)
        # if label in label_map:
        aps += [ap]
        recs += [rec[len(rec)-1]]
        print('AP for {}={:.4f}'.format(label, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('Recall ={:.3f}'.format(np.mean(recs)))

    # 检测框在图像上的可视化
    if VISUAL_FLAG:
        # 遍历所有图像
        for imagename in img_id_value_map.keys():
            # 读入该图像所有的gt框
            objs = parse_rec(os.path.join(ann_dir, imagename + ".xml"))
            R = [obj for obj in objs]  # 对于只做单一饮料输出的情况
            bbox_list = [x['bbox'] for x in R]
            cls_name = [x['name'] for x in R]  # 每个检测框类别名称
            difficult = [x['difficult'] for x in R]

            classify_miss_bboxes = []

            # 输出检测到的正确和错误的框
            bbox_right = [x['bb'] for x in classify_right_bboxes[imagename]]
            bbox_wrong = [x['bb'] for x in classify_wrong_bboxes[imagename]]
            bbox_right.extend(bbox_wrong)
            if len(bbox_right) == 0:
                continue
            if DEBUG:
                print('output bboxes %d' % len(bbox_right))
            bbox = np.array(bbox_right)

            # 遍历所有的gt框
            for bb, cls, diff in zip(bbox_list, cls_name, difficult):
                bb = np.array(bb)
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbox[:, 0], bb[0])
                iymin = np.maximum(bbox[:, 1], bb[1])
                ixmax = np.minimum(bbox[:, 2], bb[2])
                iymax = np.minimum(bbox[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bbox[:, 2] - bbox[:, 0] + 1.) *
                       (bbox[:, 3] - bbox[:, 1] + 1.) - inters)

                # 计算IoU
                overlaps = inters / uni
                # 计算最大IoU以及其对应的gt框编号
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                if ovmax > 0.5:
                    # 该gt框已经被检测到过
                    continue
                else:
                    if diff != 1:
                        # 说明该gt框没有被检测到过，为漏检
                        classify_miss_bboxes.append({'bb': bb,
                                                 'cls': cls,
                                                 'score': 1})

            if DEBUG:
                print(imagename)
                print('right: %d' % len(classify_right_bboxes[imagename]))
                print('wrong: %d' % len(classify_wrong_bboxes[imagename]))
                print('miss: %d' % len(classify_miss_bboxes))
            if len(classify_miss_bboxes) == 0:
                continue
            # print(imagename)
            # 使用opencv进行可视化
            font = cv2.FONT_HERSHEY_SIMPLEX
            image = img_id_value_map[imagename]

            # 正确分类的检测框，使用蓝色
            for bbox_dict in classify_right_bboxes[imagename]:
                bbox = bbox_dict['bb']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                output_str = '%s %.4f' % (bbox_dict['cls'], bbox_dict['score'])
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (110, 50, 50), 6)
                image = cv2.putText(image, output_str, (xmin, ymin - 5), font, 1.2, (110, 50, 50), 4)

            # 错误分类的检测框，使用红色
            for bbox_dict in classify_wrong_bboxes[imagename]:
                # if bbox_dict['score'] < 0.5:
                #     continue
                # 对于概率过小的误检，可以直接过滤
                bbox = bbox_dict['bb']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                output_str = '%s %.4f' % (bbox_dict['cls'], bbox_dict['score'])
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 6)
                image = cv2.putText(image, output_str, (xmax - 80, ymax - 5), font, 1.2, (0, 0, 255), 4)

            # 漏检的gt框，使用黄色
            for bbox_dict in classify_miss_bboxes:
                # if bbox_dict['cls'] not in label_map:
                #     print(bbox_dict['cls'])
                #     continue
                bbox = bbox_dict['bb']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                output_str = '%s' % bbox_dict['cls']
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (40, 150, 255), 6)
                image = cv2.putText(image, output_str, (xmin, ymin - 5), font, 1.2, (40, 150, 255), 4)

            # 保存图像
            img_name = os.path.join(OUTPUT_IMGS_DIR, imagename + '_add_box.jpg')
            cv2.imwrite(img_name, image)


if __name__ == "__main__":
    load_label_map(LABEL_MAP_PATH)
    print(label_map)
    ann_dir = "/data/home/zhongyisun/data/drink/"+dataset+"/Annotations"
    # ann_dir = '/home/beililiu/data/drink/drink89/Annotations'
    img_dir = "/data/home/zhongyisun/data/drink/"+dataset+"/JPEGImages"
    # img_dir = '/home/beililiu/data/drink/drink89/JPEGImages'
    test_file = "/data/home/zhongyisun/data/drink/"+dataset+"/ImageSets/Main/test.txt"
    # test_file = '/home/beililiu/data/drink/drink89/ImageSets/Main/test.txt'
    eval(img_dir, ann_dir, test_file)

