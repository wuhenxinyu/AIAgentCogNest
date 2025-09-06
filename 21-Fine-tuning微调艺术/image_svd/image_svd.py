"""
如何理解奇异值分解SVD的作用
	• 对于特征值矩阵，我们如果只包括某部分特征值，结果会怎样？
矩阵A：大小为1440*1080的图片
	• Step1，将图片转换为矩阵
	• Step2，对矩阵进行奇异值分解，得到p,s,q
	• Step3，包括特征值矩阵中的K个最大特征值，其余特征值设置为0
	• Step4，通过p,s',q得到新的矩阵A'，对比A'与A的差别
"""
import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt

# 取前k个特征，对图像进行还原
def get_image_feature(s, k):
	# 对于S，只保留前K个特征值
	s_temp = np.zeros(s.shape[0])
	s_temp[0:k] = s[0:k]
	s = s_temp * np.identity(s.shape[0])
	# 用新的s_temp，以及p,q重构A
	temp = np.dot(p,s)
	temp = np.dot(temp,q)
	plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
	plt.show()
	print(A-temp)


# 加载256色图片
image = Image.open('21-Fine-tuning微调艺术/image_svd/256.bmp') 
A = np.array(image)
# 显示原图像
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
# 对图像矩阵A进行奇异值分解，得到p,s,q
p,s,q = svd(A, full_matrices=False)
print(s)
print(len(s))
# 取前k个特征，对图像进行还原
get_image_feature(s, 5)
get_image_feature(s, 50)
get_image_feature(s, 500)
