"""
针对生成图像的后处理
	calculate_contrast() 计算图像的对比度 https://blog.csdn.net/weixin_41896508/article/details/105845969
	brightness_enhance()针对亮度的调整，考虑的是全图亮度
	brightness_enhance2()针对亮度的调整，但是考虑背景对亮度影响可能不大，所以直接对中间的缺陷区域进行调整。
	contrast_enhance()针对对比度的调整
	enhance_（）结合了前两种情况（亮度与对比度调整）并保存增强后的生成缺陷
	multi_dirs()针对某一目录下的数据增强后的批量贴回。
"""
import math
import os
import time
import numpy as np
from torchvision.transforms.functional import adjust_gamma
import cv2
from PIL import Image


def canny_mask(mask_path: str):
	"""
    传入的参数mask_path是缺陷（真实/生成）图像
    Gaussian and Canny 之后的 mask 和 有mask的图像
    :return: 可以返回一个缺陷权值矩阵mask
    """
	t = time.time()

	img_path = mask_path
	# Read the original_mask image
	img = cv2.imread(img_path, flags=0)
	print(img.shape)
	# Blur the image for better edge detection
	img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

	# Canny Edge Detection
	edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
	print(edges.shape)
	img_name = img_path.split('/')[-1]
	img_name = img_path.split('/')[-2][0:5] + img_name
	print(f"img_name:{img_name}")
	cv2.imwrite(f"./img/edges_{img_name}", edges)

	m, n = edges.shape

	index = np.where(edges == 255)
	index = np.asarray(index)

	mask = np.zeros(img.shape)
	# print(index.shape)

	for i in range(m):
		i_idx = np.where(index[0] == i)[0]
		i_idx = index[1][i_idx]
		# print(i_idx)
		if len(i_idx) >= 3:
			start = i_idx[1]
			end = i_idx[-2]
			for j in range(start, end + 1):
				mask[i][j] = 1
		elif len(i_idx) >= 1:
			start = i_idx[0]
			end = i_idx[-1]
			for j in range(start, end + 1):
				mask[i][j] = 1

	for i in range(n):
		i_idx = np.where(index[1] == i)[0]
		i_idx = index[0][i_idx]
		# print(i_idx)
		if len(i_idx) >= 3:
			start = i_idx[1]
			end = i_idx[-2]
			for j in range(start, end + 1):
				mask[j][i] = 1
		elif len(i_idx) >= 1:
			start = i_idx[0]
			end = i_idx[-1]
			for j in range(start, end + 1):
				mask[j][i] = 1
	index = np.where(mask == 1)
	index = np.asarray(index)
	mask_1 = np.zeros(mask.shape)
	# 带上了背景往里缩几个像素
	for i in range(m):
		i_idx = np.where(index[0] == i)[0]
		i_idx = index[1][i_idx]
		if len(i_idx) != 0:
			start = i_idx[0]
			end = i_idx[-1]
			for j in range(start, end + 1):
				mask_1[i][j] = 1

	# cv2.imwrite(f"./img/mask_{img_name}",mask_1*255)

	T = time.time()
	print(T - t)

	a = img * mask_1
	# cv2.imwrite(f"./img/mask_and_{img_name}",a)
	return mask_1  # masK_1 是一个0.0与1.0的array

def calculate_contrast(img_path):
	"""
	计算图像对比度的函数
	:param img_path:图像路径
	:return: 对比度值
	"""
	img0 = cv2.imread(img_path)
	print(img0.shape)
	img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)  # 彩色转为灰度图片
	m, n = img1.shape
	# 图片矩阵向外扩展一个像素
	img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	rows_ext, cols_ext = img1_ext.shape
	b = 0.0
	for i in range(1, rows_ext - 1):
		for j in range(1, cols_ext - 1):
			b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
				  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

	cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)  # 对应上面48的计算公式
	print(cg)
	return cg

def brightness_enhance(generate_mask_path, original_mask_path):
	"""
	:param generate_mask_path: 生成的缺陷路径
	:param original_mask_path: 原始缺陷路径，不是原始图片路径，主要是为了计算亮度参数gamma
	:return:亮度增强后的图像
	"""
	if generate_mask_path is None:
		generate_mask_path = "./anomaly/T6J9BF9K/single_generate_7" \
							 "19/2/2C6E9110D98378B4BA88749A8132B2E96292534_xyhw_2204_1139_167_200_0/T6J9BF9K_image_0_0.png"
	if original_mask_path is None:
		original_mask_path = "anomaly/T6J9BF9K/T6J9BF9K_masks_paddin" \
							 "g/2/2C6E9110D98378B4BA88749A8132B2E96292534_xyhw_2204_1139_167_200_0.jpg"

	generate_mask = cv2.imread(generate_mask_path)
	generate_mask_pil = Image.open(generate_mask_path)
	mean_ge = generate_mask.mean()
	original_mask = cv2.imread(original_mask_path)
	mean_ori = original_mask.mean()
	# gamma可计算或指定
	gamma = math.log(mean_ori / 255, mean_ge / 255)
	print(f"gamma:{gamma}")
	gamma = 0.3
	trans_gamma = adjust_gamma(generate_mask_pil, gamma)  # 第一个参数是PIL.Imgae
	print(f"trans_gamma:{type(trans_gamma)},{trans_gamma.size}")
	# trans_gamma.save(f"./tf/gamma/trans_gamma{gamma}.png")
	return trans_gamma


def brightness_enhance2(generate_mask_path, original_mask_path):
	"""

	:param generate_mask_path:
	:param original_mask_path:
	:return: 亮度增强后的图像以及gamma值
	"""
	# 读取图像
	generate_mask = cv2.imread(generate_mask_path)
	print(f"generate_mask:{generate_mask.shape}")
	# 得到canny算子之后的0.，1.矩阵
	erzhitu_ge = canny_mask(generate_mask_path)
	# 扩充通道
	erzhitu_ge = erzhitu_ge[:, :, None]
	if erzhitu_ge.shape[2] == 1:  # 单通道就增加维度
		erzhitu_ge = erzhitu_ge.repeat(3, axis=2)
	pixel_number_ge = erzhitu_ge.sum()  # pixel_number_ge是缺陷部分的像素数。
	erzhi_mask = erzhitu_ge * generate_mask
	mean_ge = erzhi_mask.sum() / pixel_number_ge

	original_mask = cv2.imread(original_mask_path)
	erzhitu_ori = canny_mask(original_mask_path)
	print(f"erzhi_ori.sum():{erzhitu_ori.sum()}")
	erzhitu_ori = erzhitu_ori[:, :, None]
	if erzhitu_ori.shape[2] == 1:  # 单通道就增加维度
		erzhitu_ori = erzhitu_ori.repeat(3, axis=2)
	pixel_number_ori = erzhitu_ori.sum()
	erzhi_ori = erzhitu_ori * original_mask
	mean_ori = erzhi_ori.sum() / pixel_number_ori  # 这里的pixel_number是生成缺陷并经过canny之后的像素个数。
	print(f"mean_ori:{mean_ori},mean_ge:{mean_ge}")
	gamma = math.log(mean_ori / 255, mean_ge / 255)
	# print(f"pixel_number_ge:{pixel_number_ge},pixel_number_ori:{pixel_number_ori},gamma:{gamma}")
	# gamma = 0.3
	generate_mask_pil = Image.open(generate_mask_path)
	trans_gamma = adjust_gamma(generate_mask_pil, gamma)  # 第一个参数是PIL.Imgae
	print(f"trans_gamma:{type(trans_gamma)},{trans_gamma.size}")
	# trans_gamma.save(f"./tf/enhance/gamma/trans_gamma{gamma}.png")
	return trans_gamma, gamma


def contrast_enhance(img: None, factor: int = 2.0):
	"""
	:param img:（亮度增强后的）图像
	:param factor: 对比度增强因子
	:return: 返回对比度增强后的图像
	"""
	from PIL import ImageEnhance
	generate_mask_path = "./anomaly/T6J9BF9K/single_generate_7" \
						 "19/2/2C6E9110D98378B4BA88749A8132B2E96292534_xyhw_2204_1139_167_200_0/T6J9BF9K_image_0_0.png"
	if img is None:
		generate_mask = Image.open(generate_mask_path)
	else:
		generate_mask = img
	enhancer = ImageEnhance.Contrast(generate_mask)
	# factor = 2.0 # 对比度调整参数，这个也得算。。
	img_contrasted = enhancer.enhance(factor)
	# img_contrasted.save(f"./tf/enhance/contrast/img_contrast{factor}.png")
	return img_contrasted


def enhance_(generate_mask_path, original_mask_path):
	"""
	亮度与对比度均增强后的图像
	:param generate_mask_path:
	:param original_mask_path:
	:return:
	"""
	# 先进行亮度增强
	img, gamma = brightness_enhance2(generate_mask_path, original_mask_path)
	# 计算对比度增强参数
	factor = calculate_contrast(original_mask_path) / calculate_contrast(generate_mask_path)
	# 进行对比度增强
	print(f"gamma:{gamma},factor:{factor}")
	img = contrast_enhance(img, factor)
	original_img_name = original_mask_path.split('/')[-1].split('.')[0]
	j = 0
	# 保存路径
	save_dir = f"./tf/enhance/both/{original_mask_path.split('/')[-1].split('.')[0]}/"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = save_dir + f"{original_img_name}_{j}.png"
	while os.path.exists(save_path):
		j += 1
		save_path = save_dir + f"{original_img_name}_{j}.png"
	# img.save(save_path)
	# print(f"save_path:{save_path},type(img):{type(img)}")
	return img


def multi_dirs():
	for sub_dir in os.listdir("./anomaly/T6J9BF9K/single_generate_719/1/"):
		root_path = f"anomaly/T6J9BF9K/single_generate_719/1/{sub_dir}/"
		original_mask_path = f"anomaly/T6J9BF9K/T6J9BF9K_masks_padding/1/{sub_dir}.jpg"
		# 先进行enhance
		for file in os.listdir(root_path):
			if "ep" in file:
				continue
			mask_path = root_path + file
			enhance_(mask_path, original_mask_path)
		from mask_paste_back import batch_tietu
		# 成批贴回
		mask_root_path = f"./tf/enhance/both/{sub_dir}"
		original_img_path = f"./anomaly/T6J9BF9K/trainImage/{sub_dir.split('_')[0]}.bmp"
		batch_tietu(mask_root_path, original_img_path, mode="canny")


if __name__ == "__main__":
	img_path_ge = "./anomaly/T6J9BF9K/single_generate_719/2/82D00F48753A8E34D8BCC49E840EA4586292534_xyhw_1087_99_249_49_0/T6J9BF9K_image_0_1.png"

	img_path_ori = "./anomaly/T6J9BF9K/T6J9BF9K_masks_padding/2/82D00F48753A8E34D8BCC49E840EA4586292534_xyhw_1087_99_249_49_0.jpg"

	# trans_gamma = brightness_enhance2(generate_mask_path = img_path_ge,original_mask_path = img_path_ori)
	s_t = time.time()
	img = enhance_(generate_mask_path=img_path_ge, original_mask_path=img_path_ori)  # <PIL.Image.Image>
	e_t = time.time()
	print(f"time:{e_t - s_t}")
