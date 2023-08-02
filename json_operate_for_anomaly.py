"""
    针对缺陷数据的处理：
        1）带背景的矩形框的抠出并保存：seg_extract()但是用的也是seg的外接矩形，目前没用bbox
        2）抠出segmentation并且赋白色背景并保存成矩形框mask_all（）函数，其会调用images_segs_corespondence()函数
            可以返回一个bbox/segmentation与categpry_id对应好的列表/字典。
            mask_and()函数测试是测试赋白色背景的函数
    功能函数：
        resize()缺陷并保存resize_mask
        count_if_small()计算目录下的图片大小均值等
        padding_keep_ratio():718.针对划痕等需要保持比例的缺陷图，保持比例后进行填充
        mask_all_keep_ratio():mask_all()函数保持比例版本
"""
import json
import os
import string
import numpy as np
import cv2
import shutil
import torch
import time
import shutil
import torchvision.datasets
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# 按照json文件里给的标注进行抠图，并保存,此时抠出的是矩形框，保存的也是矩形框
def seg_extract():
    ids_images = []
    ids_segmentations = []
    images_segmentations = []
    """
        with：处理json文件，将id，filename进行对应；image_id，segmentation对应；
                进而id,segmentation【bbox】对应
        处理缺陷提取：bbox的height标注是反坐标的，直接去看segmentation里的最大值和最小值，进行截取
        cv2.imread: img是先height,再width,再通道，(3648,5472,3),用一个样本【image_id == 1】实验，只有方法一可行,最后证明方法一可行
    """
    # json文件里的segmentation数据和bbox并没有完全对应
    with open("./S1ONVPHW/mark.json",encoding="utf-8") as f: #json文件需指定
        info_dict = json.load(f)
        # print(info_dict)
        print(type(info_dict))
        # 提取id与file_name关系.id与掩膜关系，file_name与segmentation关系

        for info_str,info_lists in info_dict.items():
            print(info_str)
            if info_str == "images":
                images_lists = info_lists
                for i in range(len(info_lists)):
                    id,image_name,width,height = int(images_lists[i]["id"]),images_lists[i]["file_name"],int(images_lists[i]["width"]),int(images_lists[i]["height"])
                    ids_images.append((id,image_name,width,height))
            # print(ids_images) #83张图片
        # 通过id获取segmentation
            elif info_str == "annotations":
                annotation_lists = info_lists
                for i in range(len(annotation_lists)):
                    id = int(annotation_lists[i]["image_id"])
                    segmentation = annotation_lists[i]["segmentation"]
                    category_id = annotation_lists[i]["category_id"]
                    bbox = annotation_lists[i]["bbox"]
                    ids_segmentations.append((id,bbox,category_id,segmentation))

    # 这个别放with里面
    for i in range(len(ids_images)):
        id1,image_name,width,height = ids_images[i]
        for j in range(len(ids_segmentations)):
            id2,bbox,category_id,segmentation = ids_segmentations[j]
            if id1 == id2:
                # 记录file_name,bbox,宽高,缺陷的标签,还有segmentation
                images_segmentations.append((image_name,bbox,width,height,category_id,segmentation))

    print(len(ids_images),len(ids_segmentations),len(images_segmentations))#

    dir_path = "./S1ONVPHW/testImage" # 需要指定
    n = len(images_segmentations)

    for i in range(n):
        image_name,bbox,width,height,category_id,segmentation = images_segmentations[i] # image_name:str,bbox:list,segmentaton:n*1

        # print(image_name,segmentation)
        image_path = os.path.join(dir_path,image_name)
        # bmp数据格式用Image读取

        # 1.img = cv2.imread(image_path)
        # 2.下面两行
        img = Image.open(image_path)
        img = np.asarray(img)
        print(img.shape)
        print(img.shape,bbox)

        # 将mask抠出
        points_ = np.array(segmentation[:np.newaxis]).reshape(1, -1, 2)
        points_ = points_.squeeze()

        img_ = img.copy()

        # 这样分割出来不太行，看着根本不像缺陷。。。。
        """    x_, y_, x_width, y_height = bbox[0], bbox[1], bbox[2], bbox[3]
        img_seg = img_[ y_:y_+y_height + 1,x_:x_+x_width + 1, :]
        if x_width == 0 or y_height == 0:
            continue
        else:
            print(img_seg.shape)
            save_dir = 'masks_new'
            save_path = os.path.join(save_dir, str(image_name).split('.')[-2] + '_segmentation.png')
            cv2.imwrite(save_path,img_seg)"""

        min_x,min_y,max_x,max_y = 10000,10000,0,0
        len_,_ = points_.shape
        for i in range(len_):
            x,y= points_[i][0],points_[i][1]
            if x<min_x:
                min_x = x
            if x>max_x:
                max_x = x
            if y <min_y:
                min_y = y
            if y>max_y:
                max_y = y
        if max_y > height or max_x > width:  # y 是height, x 是width
            continue

        # 第一种方法按segmentation ， 方法可行【用】
        # bmp文件格式只有两个通道
        # img_seg = img[min_y:max_y+1,min_x:max_x+1,:]
        img_seg = img[int(min_y):int(max_y)+1,int(min_x):int(max_x)+1]
        # 第二种方法按bbox1 [bbox:[3942 730 101 78] -> seg = [3942 942+730 ],[730,730+78]]
        """x_,y_,x_width,y_height = bbox[0],bbox[1],bbox[2],bbox[3]
        img_seg = img[y_:y_+y_height,x_:x_+x_width,:]"""

        # 第三种方法按bbox2 [bbox[3942 730 101 78]  -> seg = [3942 3942+730],[730-78],[730]]
        """  x_, y_, x_width, y_height = bbox[0], bbox[1], bbox[2], bbox[3]
        img_seg = img[y_ - y_height:y_+1, x_:x_ + x_width, :]"""

        print(img_seg.shape)
        save_dir = "./S1ONVPHW/masks"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        i = 0
        category_path = os.path.join(save_dir,str(category_id))
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        save_path = os.path.join(save_dir, str(category_id),str(image_name).split('.')[-2] + '_segmentation_'+str(i)+'.png')
        while os.path.isfile(save_path):
            i += 1
            save_path = os.path.join(save_dir, str(category_id),str(image_name).split('.')[-2] + '_segmentation_'+str(i)+'.png')
        cv2.imwrite(save_path, img_seg)

# 进行对应
def images_segs_corespondence(json_path):
    ids_images = []
    ids_segmentations = []
    images_segmentations = []
    """
        with：处理json文件，将id，filename进行对应；image_id，segmentation对应；
                进而id,segmentation【bbox】对应
        处理缺陷提取：bbox的height标注是反坐标的，直接去看segmentation里的最大值和最小值，进行截取
        cv2.imread: img是先height,再width,再通道，(3648,5472,3),用一个样本【image_id == 1】实验，只有方法一可行,最后证明方法一可行
    """
    # json文件里的segmentation数据和bbox并没有完全对应
    with open(json_path,encoding="utf-8") as f:
        info_dict = json.load(f)
        # print(info_dict)
        print(type(info_dict))
        # 提取id与file_name关系.id与掩膜关系，file_name与segmentation关系

        for info_str,info_lists in info_dict.items():
            print(info_str)
            if info_str == "images":
                images_lists = info_lists
                for i in range(len(info_lists)):
                    id,image_name,width,height = int(images_lists[i]["id"]),images_lists[i]["file_name"],images_lists[i]["width"],images_lists[i]["height"]
                    ids_images.append((id,image_name,width,height))
            # print(ids_images) #83张图片
        # 通过id获取segmentation
            elif info_str == "annotations":
                annotation_lists = info_lists
                for i in range(len(annotation_lists)):
                    id,segmentation,category_id = int(annotation_lists[i]["image_id"]),annotation_lists[i]["segmentation"],\
                        annotation_lists[i]["category_id"]
                    bbox = annotation_lists[i]["bbox"]
                    ids_segmentations.append((id,category_id,bbox,segmentation))

    # 这个别放with里面
    for i in range(len(ids_images)):
        id1,image_name,width,height = ids_images[i]
        for j in range(len(ids_segmentations)):
            id2,category_id,bbox,segmentation = ids_segmentations[j]
            if id1 == id2:
                images_segmentations.append((image_name,category_id,segmentation))  # 记录file_name,bbox,宽高还有segmentation

    print(len(ids_images),len(ids_segmentations),len(images_segmentations))# 83 , 301 ,332【后来运行变成301】

    dir_path = "trainImage"
    return images_segmentations # 这返回的是元组的列表，元组内容有image_name,category_id,segmentation

# 对大小不一的缺陷进行抠图并Resize成统一大小并保存
def resize_mask(path):
    mask_dir = path
    resized_mask_dir = "resized_mask_64"
    for file in os.listdir(mask_dir):
        mask_file = os.path.join(mask_dir, file)

        # 法1，直接对cv2.imread之后的ndarray进行resize
        img = cv2.imread(mask_file)
        height, width, channel = img.shape
        img_name = file.split('.')[0]
        save_path = os.path.join(resized_mask_dir, img_name + '_resized64.png')
        if height < 64 or width < 64:
            # fun1
            img_new = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        else:
            # fun2
            img_new = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_AREA)

        cv2.imwrite(save_path, img_new)

# 统计大小
def count_if_small(path):
    heights = list()
    widths = list()
    mask_dir = path
    for file in os.listdir(mask_dir):
        mask_file = os.path.join(mask_dir,file)
        img = cv2.imread(mask_file)
        height,width,channel = img.shape

        heights.append(height)
        widths.append(width)
    heights = np.array(heights)
    widths = np.array(widths)
    print("heights : ","min:",heights.min(),"max:",heights.max(),"mean:",heights.mean(),"std:",heights.std()) # 27 974 , mean:129,最小的是27，是不是resize的时候有问题
    print("widths : ","min:",widths.min(), "max:",widths.max(), "mean:",widths.mean(), "std:",widths.std()) # 28 1320 , mean:199
    return heights,widths

# 抠图并赋白色背景的一个简单示例
def mask_and_():
    # 抠多边形并赋白色背景
    img_path = "./mask_v1/trainImage/92381762B3563AEE40BB15FC6949A6383069775.jpg"
    img = cv2.imread(img_path)
    seg = [2201,711,2105,788,2029,916,2056,	985,2223,837,2319,692]

    points = np.array(seg).reshape(-1,2)
    points = points.astype(np.int64)
    # 生成points所在区域的最小外接正矩形框：左上角的x，y坐标以及矩形框的宽高
    rect = cv2.boundingRect(points)
    print(rect)
    x,y,w,h = rect
    # 将矩形框抠出
    croped = img[y:y+h,x:x+w,:].copy()
    # 使用多边形点创建遮罩
    pts = points - points.min(axis = 0) # 令矩形框左上角是0，0
    print(pts)
    mask = np.zeros(croped.shape[:2],np.uint8) # mask是二维的，不包括最后一个通道
    cv2.drawContours(mask,[pts],-1,(255,255,255),-1,cv2.LINE_AA)


    dst = cv2.bitwise_and(croped,croped,mask = mask) #除mask部分，其余都是黑的
    cv2.imshow('bitwise_and_mask_dst',dst)
    cv2.waitKey(0)

    bg = np.ones_like(croped,np.uint8)*255  # 背景纯白
    bg = cv2.bitwise_not(bg,bg,mask = mask) #
    cv2.imshow('bitwise_not',bg)
    cv2.waitKey(0)

    dst2 = bg + dst
    save_path = "padding_test.jpg"
    cv2.imwrite(save_path, dst2)
    return
# 将(第一批)缺陷数据多边形抠出并padding，均赋成白色背景/赋黑色背景
def mask_all(root_path : str = None,dataset_name : str = None):
    category_id_segs =  {}
    if dataset_name is None:
        dataset_name = "5VMQSAWN"
    json_path = root_path + f"/{dataset_name}/mark.json"
    images_segmentations = images_segs_corespondence(json_path) # 获取文件名称与seg关系
    image_path = root_path + f"/{dataset_name}/trainImage"
    save_dir = root_path + f"/{dataset_name}/{dataset_name}_masks_padding0" # 需指定
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(images_segmentations)):
        image_name ,category_id, seg = images_segmentations[i]
        sub_path = os.path.join(save_dir,str(category_id))
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        img_path = os.path.join(image_path,image_name)
        img = cv2.imread(img_path)

        points = np.array(seg).reshape(-1, 2)
        points = points.astype(np.int64)
        # 生成points所在区域的最小外接正矩形框：左上角的x，y坐标以及矩形框的宽高
        rect = cv2.boundingRect(points)
        # print(rect)
        x, y, w, h = rect

        # for category_id_segs
        if category_id not in category_id_segs:
            category_id_segs[category_id] = []
        else :
            category_id_segs[category_id].append(rect)

        # 将矩形框抠出
        croped = img[y:y + h, x:x + w, :].copy()
        # 使用多边形点创建遮罩
        pts = points - points.min(axis=0)  # 令矩形框左上角是0，0
        # print(pts)
        mask = np.zeros(croped.shape[:2], np.uint8)  # mask是二维的，不包括最后一个通道
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # dst = cv2.bitwise_and(croped, croped, mask=mask)  # 除mask部分，其余都是黑的
        dst = cv2.bitwise_and(croped, croped, mask=mask)  # 除mask部分，其余都是黑的
        # cv2.imshow('bitwise_and_mask_dst', dst)
        # cv2.waitKey(0)

        bg = np.ones_like(croped, np.uint8) * 0  # 背景纯白/纯黑，可指定*0或者*255
        if bg.shape[1] == 0 or bg.shape[0] == 0 :
            print(f"images_segs[i]:{images_segmentations[i]}")
            continue
        bg = cv2.bitwise_not(bg, bg,mask=mask)  #

        dst2 = bg + dst
        j = 0
        image_name = image_name.split('.')[0]
        save_path = os.path.join(sub_path,f"{image_name}_xyhw_{x}_{y}_{h}_{w}_{j}.jpg")
        while os.path.exists(save_path):
            j += 1
            save_path = os.path.join(sub_path,f"{image_name}_xyhw_{x}_{y}_{h}_{w}_{j}.jpg")
        cv2.imwrite(save_path, dst2)
    return category_id_segs
# 对ocr/letters数据扣除字符并保存成类

def padding_keep_ratio(img : None) ->np.ndarray :
    if img is None:
        img = cv2.imread("img.png")
    h_old,w_old,c_ = img.shape
    old_size = (h_old,w_old,c_)
    h_new,w_new,c_new = 64,64,c_
    target_size = (h_new,w_new,c_new)
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255,255,255))
    # print(f"img_new:type {type(img_new)},shape : {img_new.shape}")
    cv2.imwrite("img_keep_ratio.png",img_new)
    return img_new # return a ndarray(3) 返回的是保持比例的填充好的图像

def mask_all_keep_ratio(root_path:str = None,dataset_name:str = None):
    category_id_segs =  {}
    if dataset_name is None:
        dataset_name = "5VMQSAWN"
    json_path = root_path + f"/{dataset_name}/mark.json"
    images_segmentations = images_segs_corespondence(json_path) # 获取文件名称与seg关系
    image_path = root_path + f"/{dataset_name}/trainImage"
    save_dir = root_path + f"/{dataset_name}/{dataset_name}_masks_padding_ratio"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(images_segmentations)):
        image_name ,category_id, seg = images_segmentations[i]
        sub_path = os.path.join(save_dir,str(category_id))
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        img_path = os.path.join(image_path,image_name)
        img = cv2.imread(img_path)

        points = np.array(seg).reshape(-1, 2)
        points = points.astype(np.int64)
        # 生成points所在区域的最小外接正矩形框：左上角的x，y坐标以及矩形框的宽高
        rect = cv2.boundingRect(points)
        # print(rect)
        x, y, w, h = rect

        # for category_id_segs
        if category_id not in category_id_segs:
            category_id_segs[category_id] = []
        else :
            category_id_segs[category_id].append(rect)

        # 将矩形框抠出
        croped = img[y:y + h, x:x + w, :].copy()
        # 使用多边形点创建遮罩
        pts = points - points.min(axis=0)  # 令矩形框左上角是0，0
        # print(pts)
        mask = np.zeros(croped.shape[:2], np.uint8)  # mask是二维的，不包括最后一个通道
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # dst = cv2.bitwise_and(croped, croped, mask=mask)  # 除mask部分，其余都是黑的
        dst = cv2.bitwise_and(croped, croped, mask=mask)  # 除mask部分，其余都是黑的
        # cv2.imshow('bitwise_and_mask_dst', dst)
        # cv2.waitKey(0)

        bg = np.ones_like(croped, np.uint8) * 255  # 背景纯白/纯黑，可指定
        if bg.shape[1] == 0 or bg.shape[0] == 0 :
            print(f"images_segs[i]:{images_segmentations[i]}")
            continue
        bg = cv2.bitwise_not(bg, bg,mask=mask)  #

        dst2 = bg + dst
        dst2 = padding_keep_ratio(dst2)
        j = 0
        image_name = image_name.split('.')[0]
        save_path = os.path.join(sub_path,f"{image_name}_xyhw_{x}_{y}_{h}_{w}_{j}.jpg")
        while os.path.exists(save_path):
            j += 1
            save_path = os.path.join(sub_path,f"{image_name}_xyhw_{x}_{y}_{h}_{w}_{j}.jpg")
        print(f"save_path:{save_path},dst2:{dst2.shape}")
        cv2.imwrite(save_path, dst2)
    return category_id_segs

if __name__ == "__main__":

    root_path = "./anomaly"
    for sub_dir in os.listdir(root_path):
        if sub_dir != "T6J9BF9K": # 指定特定modelid数据
            continue
        else:
            mask_all(root_path,sub_dir)
            print(f"{sub_dir} done!")
    # dataset_dir = "./anomaly/T6J9BF9K/T6J9BF9K_masks_padding/"
    # for sub_dir in os.listdir(dataset_dir):
    #     sub_path = dataset_dir + sub_dir
    #     count_if_small(sub_path)
    padding_keep_ratio(None)
    # mask_and_()
    # category_id_segs = mask_all()
    # for i,k in category_id_segs.items():
    #     print(i,len(k))