import numpy as np
import h5py
import cv2
import os
# import scipy

# test = scipy.io.loadmat("COFW_test.mat")
# 用scio.io.loadmat读取时报错：NotImplementedError: Please use HDF reader for matlab v7.3 files
test = h5py.File('COFW_test.mat',"r")

print(test.keys())        # <KeysViewHDF5 ['#refs#', 'IsT', 'bboxesT', 'phisT']>
print(test["IsT"])        # <HDF5 dataset "IsT": shape (1, 507), type "|O">
print(test["bboxesT"])    # <HDF5 dataset "bboxesT": shape (4, 507), type "<f8">
print(test["phisT"])      # <HDF5 dataset "phisT": shape (87, 507), type "<f8">
print(type(test["IsT"]))  # <class 'h5py._hl.dataset.Dataset'>

images = np.transpose(test['IsT'])  # 所有测试集图像的ref
print(images.shape,'how many images in the dataset:507')
num = 346
img_name = images[num][0]           # 获取第 num 个图像的ref
img = test[img_name]                # 读取第 num 个图像的数据

print(img)                          # <HDF5 dataset "b": shape (239, 179), type "|u1">
print(type(img))                    # <class 'h5py._hl.dataset.Dataset'>

img = np.transpose(test[img_name])  # 转换为np数组，并对维度进行转置

print(type(img))                    # <class 'numpy.ndarray'>
print(img.shape)                    # (179, 239)

if not os.path.exists('./COFW'):
    os.mkdir('./COFW')

for i in range(images.shape[0]):
    path = os.path.join('./COFW','image'+str(i)+'.jpg')
    if i == 1:
        print(path,'check path')
    img_name = images[i][0]  # 获取第 i 个图像的ref
    # img = test[img_name]
    img = np.transpose(test[img_name])  # 转换为np数组，并对维度进行转置
    cv2.imwrite(path, img)
# save all the images

# 可视化人脸 bounding box
img = np.expand_dims(img,2).repeat(3,2)     # 将img的格式变为（w,h,3），否则cv2.rectangle会报错
print(img.shape)                            # (244, 326, 3)
boxes = test["bboxesT"]
box = boxes[:,num]
print(box)
cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[0]+box[2]),int(box[1]+box[3])), (0, 0, 255), 2)
# cv2.imshow("test_img_0_rect",img)
# cv2.waitKey(0)



# 可视化关键点及其对应的能见度
ph = np.array(test["phisT"])
x_y_v = ph[:,num]
x = x_y_v[0:29]
y = x_y_v[29:58]
v = x_y_v[58:]
for i in range(len(x)):
    temp_x, temp_y, temp_v = int(x[i]), int(y[i]), int(v[i])
    if temp_v == 1:
        cv2.circle(img, (int(temp_x),int(temp_y)), 1, (0, 0, 255), 2)
    else:
        cv2.circle(img, (int(temp_x),int(temp_y)), 1, (80, 200, 120), 2)

# cv2.imshow("test_img_0_landmark",img)
# cv2.waitKey(0)
