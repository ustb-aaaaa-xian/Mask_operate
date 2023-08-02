"""
    针对字符数据的处理：
        1）ocr_mask_extract()函数对某一个model_id传入数据以及json文件进行抠出保存
        2）ocr_multi_dirs就是对于一批几个model_id数据多次调用ocr_mask_extract()函数
    manual_class()对于药盒数据根据图片大小分别类别
    transforms_test()对于transforms里面的某些数据增强方式进行测试
    tietu() 将生成的字符/缺陷贴回到原始图像进行效果查看
    mnist/emnist数据处理：
        1）mnist_vasualize,对于mnist或者emnist数据进行可视化展示
        2）emnist_byclass_export 对于emnist数据集过大而而进行手动提取类别图片并进行保存
        3）emnist_count 统计目录下的标签数量（因为emnist数据集过大，多次处理）
        4）emnist_count 针对手动提取的emnist图片，类别数目不均衡所以每类只保留1000张，其余删掉，没有删掉，而是保存到另一个目录下了
    test_gray():此函数验证图像三个通道值一样的情况下是灰度图。

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

def ocr_mask_extract(dataset_dir,json_dir,save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"save_dir does not exsit.os mkdirs.")
    ids_images = []
    ids_bboxes_categories = []
    imagename_bboxes_categories = []
    with open(json_dir, encoding="utf-8") as f:
        info_dict = json.load(f)
        for info_str,info_lists in info_dict.items():
            if info_str == "images":
                images_lists = info_lists
                for i in range(len(info_lists)):
                    id, image_name, width, height = int(images_lists[i]["id"]), images_lists[i]["file_name"], \
                    images_lists[i]["width"], images_lists[i]["height"]
                    ids_images.append((id, image_name, width, height))
        for info_str,info_lists in info_dict.items():
            if info_str == "annotations":
                annotation_lists = info_lists
                for i in range(len(info_lists)):
                    bbox,image_id,category_name = annotation_lists[i]["bbox"],int(annotation_lists[i]["image_id"]),\
                        annotation_lists[i]["category_name"]
                    ids_bboxes_categories.append((image_id,bbox,category_name))
        # 进行对应
        for i in range(len(ids_bboxes_categories)):
            image_id1,bbox,category_name = ids_bboxes_categories[i]
            for j in range(len(ids_images)):
                image_id2,image_name,width,height = ids_images[j]
                if image_id1 == image_id2:
                    imagename_bboxes_categories.append((image_name,bbox,category_name,width,height))
        print(imagename_bboxes_categories)
        # get ids_boses_categories
        # print(imagename_bboxes_categories)
        # classes = ['0','1','2','3','4','5','6','7','8','9','/',':','*','?','"','<','>','|'] # 缺一个反斜杠'\'
        classes = string.digits + string.ascii_lowercase +string.ascii_uppercase # 不能用[]，加了[]之后就变成列表而不是字符串了
        print(classes)
        # string.digits
        for i in range(len(imagename_bboxes_categories)):
            image_name,bbox,category_name,width,height = imagename_bboxes_categories[i]

            if category_name  not in classes: # 根据自己需要去看取哪些类别，字母，数字还是混合
                continue
            # 对于testImage,好像并没有标注。
            if image_name not in os.listdir(dataset_dir):
                continue
            image_path = os.path.join(dataset_dir,image_name)
            img = cv2.imread(image_path) # height width  channel
            x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
            # ocr_img = img[y:y+h,x:x+w,:].copy()
            ocr_img = img[y:y+h,x:x+w,:]
            if y+h>=height or x+w>=width:
                print(f"{image_name}'s category_name{category_name} and bbox{bbox} wrong!!!!,width ,height ={width,height} but y+h and x+w is {y+h,x+w}.")
                continue
            ocr_category = category_name   #
            path = os.path.join(save_dir,ocr_category)
            if not os.path.exists(path):
                os.makedirs(path)
            j = 0
            image_name = image_name.split(".")[0] # 之前可能还会用到image_name的路径，所以
            save_path = os.path.join(path,f"{image_name}_{ocr_category}_{j}.jpg")
            while os.path.exists(save_path):
                j += 1
                save_path = os.path.join(path,f"{image_name}_{ocr_category}_{j}.jpg")
            cv2.imwrite(save_path,ocr_img)

# 对某一目录下的所有类别均进行字符抠出并保存
def ocr_mask_multi_dirs():
    # 对新一批ocr进行字符抠出，
    ocr2_path = "ocr4"
    for sub_dir in os.listdir(ocr2_path):
        if sub_dir == "GGASVUPO" or sub_dir == "HDFXTAA7":
            continue
        trainimg_path = os.path.join(ocr2_path,sub_dir,"trainImage")
        json_path = os.path.join(ocr2_path,sub_dir,"mark.json")
        save_dir = os.path.join(ocr2_path,sub_dir,f"{sub_dir}_ocr_all")
        # 已经进行处理了,直接注释掉
        ocr_mask_extract(trainimg_path,json_path,save_dir)
        print(f"{ocr2_path}/{sub_dir}_done!")

# 将药盒类别的奇怪类别挑出，并单独组成新的类
def  manual_classes(dataset_dir):
    for sub_dir in os.listdir(dataset_dir):
        sub_path = os.path.join(dataset_dir,sub_dir)
        for img in os.listdir(sub_path):
            img_path = os.path.join(dataset_dir,sub_dir,img)
            img_name = img.split(".")[0]
            img_ = cv2.imread(img_path)
            h,w,c = img_.shape
            if h>100 or w>100:
                src = img_path
                dst_dir = os.path.join(dataset_dir,f"{sub_dir}_")
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                i = 0
                dst = os.path.join(dst_dir,f"{img_name}_copy_{i}.png")
                while os.path.exists(dst):
                    i += 1
                    dst = os.path.join(dst_dir,f"{img_name}_copy_{i}.png")
                shutil.move(src,dst)

# 测试多种transforms函数作用进行数据增强
def transforms_test():
    # 随机的数据增强
    img_path = "./S1ONVPHW/S1ONVPHW_masks/1/3D978E8754978B1DEC00FB078919DCA46292534_segmentation_0.png"# mean:tensor(0.7379) min:tensor(0.2078) max:tensor(0.9451) std:tensor(0.2076)
    # img_path2 = "ocr3/Q8CBR25P/Q8CBR25P_patch_ocr/3/0A7529722831C5392A1ABCDEEA85E5873314527_3_0.png"# mean :tensor(0.2997) min:tensor(0.0588) max:tensor(0.7725) std:tensor(0.1688)

    image_name = img_path.split("/")[-1].split('.')[0]
    print(image_name)
    tf_original = transforms.Compose(
        [
            # transforms.Grayscale(3),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomRotation(degrees = 25),
            transforms.Resize((28,28)),
            transforms.ToTensor(),

        ]
    )
    tf_original2 = transforms.Compose(
        [
            transforms.RandomRotation(degrees=5,interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    # 先把旋转去掉，可能会漏边,randomcrop不行，因为数据本身就占了整个字符，不能再继续中心裁剪
    tf1 = transforms.RandomAffine(degrees=15)
    tf_pad = transforms.Pad(padding = 20,fill = 1)
    # tf2 = transforms.RandomRotation(degrees = 15)
    tf3 = transforms.GaussianBlur(kernel_size=5, sigma=(.1, .2))
    tf4 = transforms.GaussianBlur(kernel_size=3, sigma=(.1, .1))
    # 加入以下几种，
    tf_1 = transforms.RandomApply([tf1, tf3], p=0.5)
    tf_2 = transforms.RandomApply([tf1, tf4], p=0.5)
    tf_3 = transforms.RandomApply([tf3, tf4], p=1)
    tf_randomcrop = transforms.RandomCrop(size = 25,padding = 3,pad_if_needed= True,fill = 1)
    tfs = [tf1,tf3,tf4,tf_1,tf_2,tf_3]
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x :x.repeat(3,1,1))
        ]
    )


    """
        测试对比度增强的感觉
    """
    img = Image.open(img_path)
    img.save(f"./tf/{image_name}_original.png")
    print(f"image.shape:{img.size}")
    # print(img_contrast.mean(),img_contrast.min(),img_contrast.max()) # numpy形式可以这么写，其实转成Tensor之后也可以这么写
    img_ = tf(img)
    # Image与Tensor的保存方式不同
    # img_.save(f"./tf/{image_name}_contrast.png")
    print(img_.shape)
    save_image(img_,f"./tf/{image_name}_contrast.png")

def mean_std():
    dir_path = "./T6J9BF9K/T6J9BF9K_masks_padding_manual/3/"
    means,stds = [],[]
    for file in os.listdir(dir_path):
        img_path = os.path.join(dir_path,file)
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        print(f"img_tensor:shape,mean,std:{img_tensor.shape},{img_tensor[0,:,:].mean(),img_tensor[1,:,:].mean(),img_tensor[2,:,:].mean()},"
              f"{img_tensor[0,:,:].std(),img_tensor[1,:,:].std(),img_tensor[2,:,:].std()}")
        means.append(img_tensor[0,:,:].mean().item())
        stds.append(img_tensor[0,:,:].std().item())
    print(f"means.mead(),stds.mean():{np.asarray(means).mean(),np.asarray(stds).mean()}")

# 对MNIST/EMNIST图像的可视化，EMNIST数据集都要transpose才变正
def mnist_visualize():
    import matplotlib.pyplot as plt
    # test_ds = torchvision.datasets.MNIST(root = "./data",train = False)
    test_ds = torchvision.datasets.EMNIST(root = "./data",split = "letters",train = False)
    ds_images, ds_targets = (test_ds._load_data())
    print(f"len of emist_letters:{len(ds_targets)}")
    fig = plt.figure()  # 生成图框

    for i, c in enumerate(np.random.randint(0, 10000, 25)):  # 随机取0，1000里的25张图片

        plt.subplot(5, 5, i + 1)

        plt.tight_layout()  # 调整间距
        t_ = ds_images[c].transpose(0,1)
        # t_ = ds_images[c]
        # plt.imshow(ds_images[c], interpolation='none')
        # print(ds_images.type,ds_images[c].shape)
        # plt.title("数字标签: {}".format(ds_targets[c]))
        plt.imshow(t_, interpolation='none')
        # print(ds_images.type, t_.shape)
        plt.title("数字标签: {}".format(ds_targets[c]))
        plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.show()

# 对于EMNIST数据集总是记载不出来的问题，直接进行手动图片保存
def emnist_byclass_export():
    import matplotlib.pyplot as plt
    test_ds = torchvision.datasets.EMNIST(root = "./data",split = "byclass",train = True)
    ds_images, ds_targets = (test_ds._load_data())
    print(f"ds_image.type:{type(ds_images)},ds_images.shape:{ds_images.shape},ds_targets.type:{type(ds_targets)},ds_targets.shape:{ds_targets.shape}")
    # [697932, 28, 28] [697932]
    dst_data_path = "./data/emnist_byclass_manual/"
    if not os.path.exists(dst_data_path):
        os.makedirs(dst_data_path)
    time1 = time.time()
    for i in range(200000,300000): #分几次弄，range里面数量不一样
        # tensor ->numpy ->image
        if i %1000 == 0:
            time2 = time.time()
            print(f"i = {i},{time2-time1}s has passed after 1000 characters.")
            time1 = time2
        target = str(ds_targets[i])
        data = ds_images[i].reshape(28,28).transpose(0,1)
        data = data.numpy()
        image = Image.fromarray(data)
        target_path = dst_data_path + str(target)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        j = len(os.listdir(target_path))
        save_path = dst_data_path + str(target)+f"/{str(target)}_str{j}.png"
        while os.path.exists(save_path):
            j += 1
            save_path = dst_data_path + str(target)+f"/{str(target)}_str{j}.png"
        image.save(save_path)

# 上一个手动图片保存的函数，数据集过大，后半段没执行完，所以统计一下图片数量
def emnist_count():
    dir = "./data/emnist_byclass_manual"
    len_all = 0
    lens = {}
    for sub_dir in os.listdir(dir):
        len_ = len(os.listdir(os.path.join(dir,sub_dir)))
        len_all += len_
        lens[sub_dir] = len_
    print(f"lens:{lens}")
    print(len_all)
    return lens

# 对于手动提取的emnist_byclass数据，数据量不均衡，目前打算仅使用每类1000个数据，所以直接将每类1000个弄成新的文件夹。
def delete_redundancy():
    source_dir = "./data/emnist_byclass_manual"
    dst_dir = "./data/emnist_byclass_balanced1000"
    for sub_dir in os.listdir(source_dir):
        sub_path = os.path.join(source_dir,sub_dir)
        dst_sub_dir = os.path.join(dst_dir,sub_dir)
        if not os.path.exists(dst_sub_dir):
            os.makedirs(dst_sub_dir)
        j = 0
        for file in os.listdir(sub_path):
            if j>=1000:
                break
            file_path = os.path.join(sub_path,file)
            shutil.copy(file_path,dst_sub_dir)
            j += 1

def test_gray():
    # 以下程序验证三个通道一样是灰度图！！！
    img = cv2.imread("img.png")
    img_ = np.ones_like(img)
    for i in range(3):
        if i == 0:
            img_[:,:,i] = img[:,:,0]
        elif i == 1:
            img_[:,:,i] = (img[:,:,0]+5) % 255
        elif i == 2:
            img_[:,:,i] = (img[:,:,0] +10) % 255
    cv2.imwrite("./tf/img_2_.png",img_)
if __name__ == "__main__":

    # 手动挑出药盒上的其余类别
    # manual_classes("./ocr2/G716DVBX/G716DVBX_patch_ocr")

    # 6.26 挑字母 ，用ocr_mask_extract函数，里面记录了所需要的类别，现在只进行字母的跳出，所以直接
    dataset_name = "QJCXDX0Y"
    # ocr_mask_extract(dataset_dir=f"./model0627_letters/{dataset_name}/trainImage",json_dir=f"./model0627_letters/{dataset_name}/mark.json",save_dir = f"./model0627_letters/{dataset_name}/{dataset_name}_patch_letters")
    # mnist_visualize()
    # emnist_count() # 第一次到75323 ，第二次77277, 第三次80000,100000,150000
    # delete_redundancy() #已经完成
    # count_if_small("S1ONVPHW/S1ONVPHW_masks/1")
    test_gray()
