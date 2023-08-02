"""
贴图函数(还有其他函数)：
    三个回帖函数的选择：
        gaussian_2d()生成二维的高斯函数 https://blog.csdn.net/weixin_39714307/article/details/110325903
        canny_mask():from_luodi，抠出缺陷并生成二值图像的函数
        canny_and_gaussian():针对生成的缺陷数据，先进行canny算子得到01矩阵，再放进gaussian得到蒙版高斯，其实就是背景
    tietu()最基本的覆盖
    tietu_padding()针对padding生成的数据进行贴回，涉及图片融合操作，指定图片途径，缺陷路径，以及位置参数h_top以及w_left。还有参数mode,代表了是怎么进行融合的，可以在命名的适合方便认。
    tietu_multi()：给定mask_paths即缺陷的路径集合，以及给定image_paths()即图片的集合，在贴的时候可随机选取缺陷，以及随机选取被贴图片，位置的选择（在函数里写了可以指定原始抠出位置，也可自己指定）
    location_test():针对老板指定的图片，找到板的位置，沿边贴，此函数就是找到板的height,width边界
    batch_tietu():传入参数：1）生成的缺陷的路径，此文件名带有xyhw,所以有原缺陷位置。2）要贴回的图像路径
"""
import os
import random

import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
from json_operate_for_anomaly import mask_all
from postprocess import enhance_
def location_test():
    dataset_name = "5VMQSAWN"
    img_path = "anomaly/5VMQSAWN/trainImage/090DB746B1D51273ECF318BC15E27AF06656040.jpg"
    img = cv2.imread(img_path)
    h,w = 1150,0
    sub_img = img[h:h+1600,w:w+2700,:].copy()
    sub_path = f"./img/{dataset_name}"
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    cv2.imwrite(f"./img/{dataset_name}/sub_img.png",sub_img)
def gaussian_2d(h:int,w:int):
    # h = 64
    # w = 64
    center_x = w // 2
    center_y = h // 2
    R = np.sqrt(center_x **2 + center_y ** 2)
    gaussian_map = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            dis = np.sqrt((i-center_x) ** 2 + (j-center_y) ** 2)
            gaussian_map[i][j] = np.exp(-1. * dis /R)
    # print(gaussian_map)
    # gaussian_map = gaussian_map[:,:,None]
    # gaussian_map.repeat(3,axis = 2)
    print(f"gaussian_map.shape:{gaussian_map.shape}")
    # cv2.imshow("gaussian_liangdu",(gaussian_map*255).astype(np.uint8))
    return gaussian_map

def canny_mask(mask_path: str ):
    """
    传入的参数mask_path是缺陷（真实/生成）图像
    Gaussian and Canny 之后的 mask 和 有mask的图像
    :return: 可以返回一个缺陷权值矩阵mask
    """
    t=time.time()

    img_path = mask_path
    # Read the original_mask image
    img = cv2.imread(img_path,flags=0)
    print(img.shape)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    print(edges.shape)
    img_name = img_path.split('/')[-1]
    img_name = img_path.split('/')[-2][0:5] +img_name
    print(f"img_name:{img_name}")
    cv2.imwrite(f"./img/edges_{img_name}",edges)

    m,n = edges.shape

    index=np.where(edges==255)
    index=np.asarray(index)

    mask=np.zeros(img.shape)
    #print(index.shape)

    for i in range(m):
        i_idx=np.where(index[0]==i)[0]
        i_idx=index[1][i_idx]
        #print(i_idx)
        if len(i_idx)>=3:
            start=i_idx[1]
            end=i_idx[-2]
            for j in range(start,end+1):
                mask[i][j]=1
        elif len(i_idx) >=1:
            start = i_idx[0]
            end = i_idx[-1]
            for j in range(start, end + 1):
                mask[i][j] = 1

    for i in range(n):
        i_idx=np.where(index[1]==i)[0]
        i_idx=index[0][i_idx]
        #print(i_idx)
        if len(i_idx)>=3:
            start=i_idx[1]
            end=i_idx[-2]
            for j in range(start,end+1):
                mask[j][i]=1
        elif len(i_idx) >=1 :
            start = i_idx[0]
            end = i_idx[-1]
            for j in range(start, end + 1):
                mask[j][i] = 1
    index=np.where(mask==1)
    index=np.asarray(index)
    mask_1=np.zeros(mask.shape)
    # 带上了背景往里缩几个像素
    for i in range(m):
        i_idx=np.where(index[0]==i)[0]
        i_idx=index[1][i_idx]
        if len(i_idx)!=0:
            start=i_idx[0]
            end=i_idx[-1]
            for j in range(start,end+1):
                mask_1[i][j]=1

    # cv2.imwrite(f"./img/mask_{img_name}",mask_1*255)

    T=time.time()
    print(T-t)

    a = img * mask_1
    # cv2.imwrite(f"./img/mask_and_{img_name}",a)
    return mask_1 # masK_1 是一个0.0与1.0的array

# 先用canny算子得到0.,1.矩阵，然后再蒙版高斯
def canny_and_gaussian(mask_path:str = None):
    canny_matrix = canny_mask(mask_path)
    h,w = canny_matrix.shape[0],canny_matrix.shape[1]
    center_x = w // 2
    center_y = h // 2
    R = np.sqrt(center_x **2 + center_y ** 2)
    # gaussian_map = np.zeros((h,w))
    gaussian_map = canny_matrix
    for i in range(h):
        for j in range(w):
            dis = np.sqrt((i-center_x) ** 2 + (j-center_y) ** 2)
            if gaussian_map[i][j] != 0:
                gaussian_map[i][j] = np.exp(-1. * dis /R)
            else:
                continue
    # print(gaussian_map)
    print(f"gaussian_map.shape:{gaussian_map.shape}")
    return gaussian_map
def tietu():
    # 原始图像

    original_image_path = "./T6J9BF9K/trainImage/0BC0254A56EB6214F4969E21CF39AA3B6292534.bmp"  # 大概400*400
    original_file_name = original_image_path.split('/')[-1].split('.')[0]
    original_image = cv2.imread(original_image_path)
    h_max, w_max, _ = original_image.shape
    print(f"original_image shape :{original_image.shape}")

    # 生成的缺陷图像
    generate_mask_path = "./T6J9BF9K/masks_T6J9BF9K_cifar64_padding_200/T6J9BF9K_image_w0.0_0_1.png"
    tf = transforms.Resize((61, 65))
    mask_image = tf(Image.open(generate_mask_path))
    mask_image = np.asarray(mask_image)
    # mask_image = cv2.imread(generate_mask_path)
    print(f"mask_image.shape:{mask_image.shape}")
    mask_image_name = generate_mask_path.split('/')[-1].split('.')[:-1]
    h, w, c = mask_image.shape
    image = original_image.copy()
    w_left = w_max // 2
    h_top = 330
    image[h_top:h_top + h, w_left:w_left + w, :] = mask_image
    cv2.imwrite(f'./tf/{original_file_name}_{mask_image_name}.png', image)
    w_left = 0
    """
    ocr_image_dir = "./generate_save/ocr1_3_channels/P0X8CM8Q_1/"
    for file in os.listdir(ocr_image_dir):
        ocr_image_path = os.path.join(ocr_image_dir,file)
        ocr_image = cv2.imread(ocr_image_path)
        h, w, c = ocr_image.shape
        w_right = w_left + w
        if w_right >= w_max:
            break
        else:
            # original_image[h:2*h,w_left:w_left+w,:] = ocr_image  # 每一次都在已经被处理过的original_image上进行，所以才会被贴图多次
            image = original_image.copy()
            image[h:2*h,w_left:w_left+w,:] = ocr_image
            w_left = w_right
            cv2.imwrite(f'./tf/{original_file_name}_{file}.png', image)
            continue"""

    # ocr_image = cv2.imread(ocr_image_path)
    # h,w,c = ocr_image.shape
    # print(h,w,c)

    # 先贴左上角
def tietu_padding(original_image_path,generate_mask_path,original_mask_path,alpha_matrix,h_top,w_left,mode):
    """
    :param,首先有一个原始图像路径，然后有一个生成的缺陷图像，再之后就是权值矩阵alpha_matrix,还有
    :return:进行贴回图像的写入
        比较重点的是保存路径的设置save_dir.
    """
    original_file_name = original_image_path.split('/')[-1].split('.')[0] # 得到文件名称
    original_image = cv2.imread(original_image_path) # ndarray图像
    h_max, w_max, _ = original_image.shape # 原始图像的高宽通道
    print(f"original_image shape :{original_image.shape}")

    m,n = alpha_matrix.shape # （canny）算子得到的alpha_matrix


    mask_image = enhance_(generate_mask_path,original_mask_path) # an img
    tf = transforms.Compose([
        transforms.Resize((m,n)),
        transforms.ToTensor()])
    mask_image = tf(mask_image) #将生成图像进行resize以和原始缺陷大小一致，如果形状一致即转换为Tensor（相撞转换可以）
    mask_image = mask_image.numpy() * 255 # 从0-1tensor变0-255np.uint8


    mask_image = np.transpose(mask_image,(1,2,0)).astype(np.uint8) # tensor形状是c h w 转换为 ndarray的 h w c


    print(f"mask_image.shape:{mask_image.shape},type(mask_image):{type(mask_image)}")

    h, w, c = mask_image.shape


    # 开始回贴
    image = original_image.copy()

    original_mask_part = image[h_top:h_top + h, w_left:w_left + w, :].copy() #原始图像的要贴回部分

    # 进行权重分配
    if len(alpha_matrix.shape) == 2:
        alpha_matrix = alpha_matrix[:,:,None] #增加维度
    if alpha_matrix.shape[2] == 1: # 单通道就增加维度
        alpha_matrix.repeat(3,axis = 2)


    # 直接加和操作覆盖原图
    mask_image = (mask_image * alpha_matrix).astype(np.uint8)+(original_mask_part*(1-alpha_matrix)).astype(np.uint8)
    image[h_top:h_top + h, w_left:w_left + w, :] = mask_image
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)


    # 开始进行保存
    dataset_name = "T6J9BF9K"
    # 保存路径，可以指定
    save_dir = f"./tf/{dataset_name}/{dataset_name}_801_modes_c2/{original_file_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    j = 0
    save_path = save_dir + f"{original_file_name}_{mode}_{j}.png"
    while os.path.exists(save_path):
        j += 1
        save_path = save_dir + f"{original_file_name}_{mode}_{j}.png"
    cv2.imwrite(save_path, image)
    print("Done")

# 可以将缺陷路径弄个列表，图片路径弄个列表，缺陷位置也是随机选。。真就全部随机，但是要保证alpha_matrix大小所以不再对大小进行调整
def tietu_multi():
    mask_paths = []
    img_paths = []
    mask_root_path = "./anomaly/5VMQSAWN/5VMQSAWN_c_0/"
    for file in os.listdir(mask_root_path):
        mask_path = f"{mask_root_path}" + file
        mask_paths.append(mask_path)
    image_root_path = "./anomaly/5VMQSAWN/for_paste/"
    for file in os.listdir(image_root_path):
        image_path = f"{image_root_path}" + file
        img_paths.append(image_path)

    category_id_segs = mask_all(root_path="./anomaly", dataset_name="5VMQSAWN")

    for i in range(50):
        i_mask = random.randint(0, len(mask_paths)-1)
        i_image = random.randint(0, len(img_paths)-1)
        mask_path = mask_paths[i_mask]

        # image_path = "anomaly/5VMQSAWN/trainImage/090DB746B1D51273ECF318BC15E27AF06656040.jpg"
        image_path = img_paths[i_image]
        s_t = time.time()
        alpha_matrix = canny_mask(mask_path)
        tietu_padding(image_path,mask_path,alpha_matrix)
        e_t = time.time()
        print(f"{e_t-s_t}!" , end = "\r")



# 对于一整个数据集进行贴图，贴图范围为原始缺陷的周围，给定参数为,mask_root_path,original_img_path
def batch_tietu(mask_root_path,original_img_path,mode):
    """
    给定生成的缺陷图像（名称里包含原始缺陷位置信息），原始图像路径，以及贴回算法
    :param mask_root_path:
    :param original_img_path:
    :param mode:
    :return:
    """
    mask_paths = []
    category_id_segs = []

    for file in os.listdir(mask_root_path):
        # 只需要生成的单个缺陷，不需要拼接图像以及权值文件
        if "ep" in file or "pth" in file :
            continue
        else:
            mask_path = f"{mask_root_path}/" + file
            mask_paths.append(mask_path)
    if len(mask_paths) == 0: #判断列表内是否有文件，如果无则返回
        return
    # 找到原缺陷位置,位置信息隐含在目录名称里。
    locations = mask_root_path.split('/')[-1].split('_')[2:]
    x_,y_,h_,w_ = int(locations[0]),int(locations[1]),int(locations[2]),int(locations[3])
    # 得到原始图像的高宽
    h_max,w_max = cv2.imread(original_img_path).shape[0],cv2.imread(original_img_path).shape[1]

    len_ = max(h_,w_) # len_代表缺陷宽高的较大值
    bianjies = [len_,len_ * 2] # 目前贴在原缺陷周围

    # 一个新的参数，原始缺陷的路径（需要传给特征增强函数）
    dataset_name = "T6J9BF9K"
    original_mask_path = f"./anomaly/{dataset_name}/{dataset_name}_masks_padding0/2/{mask_root_path.split('/')[-1]}" + ".jpg"
    for j in range(len(bianjies)):
        bianjie = bianjies[j]
        # 确定左右上下边界
        w_left_ = 0 if (x_ - bianjie) <= 0 else (x_ - bianjie)
        w_right_ = w_max  if (x_ +bianjie)>=w_max else (x_ + bianjie)
        h_top_ = 0 if (y_ -bianjie) <= 0 else (y_ - bianjie)
        h_bottom_ = h_max  if (y_+bianjie)>=h_max else (y_ + bianjie)
        print(f"w_left_:{w_left_},w_right_:{w_right_},h_top_:{h_top_},h_bottom_:{h_bottom_}")

        # 左边的从上到下
        h_top = h_top_
        w_left = w_left_
        while h_top <= h_bottom_ - len_:
            i_mask = random.randint(0, len(mask_paths) - 1)
            mask_path = mask_paths[i_mask]
            h,w = h_,w_
            if mode == "canny":
                alpha_matrix = canny_mask(mask_path)
            elif mode == "gau":
                alpha_matrix = gaussian_2d(h,w)
            elif mode == "both":
                alpha_matrix = canny_and_gaussian(mask_path)
            else:
                alpha_matrix = canny_and_gaussian(mask_path)
            tietu_padding(original_img_path,mask_path,original_mask_path,alpha_matrix,h_top,w_left,mode)
            h_top = h_top + int(len_ * 1.5)
        print(f"左边的从上到下 done!")

        # 右边的从上到下
        h_top = h_top_  # 最上面
        w_left = w_right_ if w_right_ <= w_max - len_ else w_max - len_
        while h_top <= h_bottom_ - len_:
            i_mask = random.randint(0, len(mask_paths) - 1)
            mask_path = mask_paths[i_mask]
            h, w = h_,w_
            if mode == "canny":
                alpha_matrix = canny_mask(mask_path)
            elif mode == "gau":
                alpha_matrix = gaussian_2d(h,w)
            elif mode == "both":
                alpha_matrix = canny_and_gaussian(mask_path)
            else:
                alpha_matrix = canny_and_gaussian(mask_path)
            tietu_padding(original_img_path, mask_path,original_mask_path, alpha_matrix,  h_top, w_left,mode)
            h_top = h_top + int(len_ * 1.5)
        print(f"右边的从上到下 done!")

        # 上边的从左到右
        h_top = h_top_
        w_left = w_left_
        while w_left <= w_right_ - len_:
            i_mask = random.randint(0, len(mask_paths) - 1)
            mask_path = mask_paths[i_mask]
            h, w = h_,w_
            if mode == "canny":
                alpha_matrix = canny_mask(mask_path)
            elif mode == "gau":
                alpha_matrix = gaussian_2d(h, w)
            elif mode == "both":
                alpha_matrix = canny_and_gaussian(mask_path)
            else:
                alpha_matrix = canny_and_gaussian(mask_path)
            tietu_padding(original_img_path, mask_path, original_mask_path,alpha_matrix,  h_top, w_left,mode)
            w_left = w_left + int(len_ * 1.5)
        print(f"上边的从左到右 done!")

        # 下边的从左到右
        h_top = h_bottom_ if h_bottom_ <= h_max - len_ else h_max - len_  # 最下面
        w_left = w_left_

        while w_left <= w_right_ - len_:
            i_mask = random.randint(0, len(mask_paths) - 1)
            mask_path = mask_paths[i_mask]
            # 不同的alpha_matrix的选择
            h, w = h_,w_
            if mode == "canny":
                alpha_matrix = canny_mask(mask_path)
            elif mode == "gau":
                alpha_matrix = gaussian_2d(h,w)
            elif mode == "both":
                alpha_matrix = canny_and_gaussian(mask_path)
            else:
                alpha_matrix = canny_and_gaussian(mask_path)
            tietu_padding(original_img_path, mask_path,original_mask_path, alpha_matrix, h_top, w_left,mode)
            w_left = w_left + int(len_ * 1.5)
        print(f"下边的从左到右 done!")


if __name__ == "__main__":
    # canny_mask("anomaly/5VMQSAWN/single_generate_2/81D80D2356367F33FD28F1CE6524156B7000618_xyhw_1128_3123_82_79_0/5VMQSAWN_image_0_0.png")
    dataset_name = "T6J9BF9K"
    root_path = f"./anomaly/{dataset_name}/single_generate_731/2/" # 对应生成的图片的位置
    for sub_dir in os.listdir(root_path):
        print(f"sub_dir:{sub_dir}")
        s_t = time.time()
        mask_paths = []
        mask_root_path = root_path + sub_dir
        original_img_name = sub_dir.split('_')[0] + ".bmp"  # 后缀有的是bpm有的是jpg，需要指定一下
        original_img_path = f"./anomaly/{dataset_name}/trainImage/{original_img_name}"
        modes = ["gau","canny","both"]
        for mode in modes:
            if mode != "canny":
                continue
            batch_tietu(mask_root_path,original_img_path,mode = mode) # mode 代表贴回使用的算法，gau还是canny还是both
        e_t = time.time()
        print(f"sub_dir:{sub_dir} done! {e_t - s_t} has passed after three modes being applied!")