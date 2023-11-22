# import numpy as np
# from PIL import Image
# import os
# from torchvision import transforms
# import cv2
# import torch
#
#
#
# def inverse_depth_norm(depth):
#     depth = maxDepth / depth
#     depth = torch.clamp(depth, maxDepth / 100, maxDepth)
#     return depth
#
#
#
# test_rgb_path = "/HOMES/yigao/KITTI/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/"
# files = [ f for f in os.listdir(test_rgb_path) if '.jpg' in f.lower() ]
# sorted_rgb_files = sorted(files, key=lambda x: int(float(x.split('.')[0].split('_')[1])))
#
#
# test_depthGT_path = "/HOMES/yigao/KITTI/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/"
# files = [ f for f in os.listdir(test_depthGT_path) if '.png' in f.lower() ]
# sorted_depthGT_files = sorted(files, key=lambda x: int(float(x.split('.')[0].split('_')[1])))
# # print(sorted_depthGT_files)
#
#
# trans = transforms.Compose([transforms.Resize(size=(384, 1280))])
#
# # output_loc = "/HOMES/yigao/KITTI/vkitti_testset"      # for GDM
# # output_loc = "/HOMES/yigao/KITTI/vkitti_testset_test/test"      # for GDM test original
# output_loc = "/HOMES/yigao/KITTI/test"      # for GDM
# count = 1
#
# maxDepth = 80
#
#
# for rgb, depthgt in zip(sorted_rgb_files, sorted_depthGT_files):
#     # print(rgb)
#     # print(depthgt)
#     rgb_img = Image.open( test_rgb_path + rgb)#.convert('RGB')           #an additional alpha channel per pixel https://stackoverflow.com/questions/58496858/pytorch-runtimeerror-the-size-of-tensor-a-4-must-match-the-size-of-tensor-b
#     depth_img = Image.open(test_depthGT_path + depthgt)
#     # print(rgb_img)
#     rgb_img = trans(rgb_img)
#     depth_img = trans(depth_img)
#
#     rgb_img = np.asarray(rgb_img)
#     depth_img = np.asarray(depth_img)
#     # print(rgb_img.shape)
#     print(depth_img)
#     # print(depth_img)
#     # if not os.path.exists(output_loc + "/%#05d" % (count)):
#     #     os.mkdir(output_loc + "/%#05d" % (count))
#     # np.save(output_loc + "/%#05d" % (count) + "/image" + "%#05d" % (count), rgb_img)
#     # np.save(output_loc + "/%#05d" % (count) + "/depth" + "%#05d" % (count), depth_img)
#
#     # depth_img = cv2.normalize(depth_img, depth_img, 0.8, 80, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     buffer = np.copy(depth_img)
#     buffer = torch.from_numpy(buffer)
#     depth_img = inverse_depth_norm(buffer)
#     # print(depth_img)
#     # np.savez_compressed(output_loc + "/%#05d" % (count), image=rgb_img, depth=depth_img)
#
#     count += 1





import numpy as np
from PIL import Image
import os
from torchvision import transforms
import cv2
import torch




test_rgb_path = "/HOMES/yigao/KITTI/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/"
files = [ f for f in os.listdir(test_rgb_path) if '.jpg' in f.lower() ]
sorted_rgb_files = sorted(files, key=lambda x: int(float(x.split('.')[0].split('_')[1])))


test_depthGT_path = "/HOMES/yigao/KITTI/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/"
files = [ f for f in os.listdir(test_depthGT_path) if '.png' in f.lower() ]
sorted_depthGT_files = sorted(files, key=lambda x: int(float(x.split('.')[0].split('_')[1])))
# print(sorted_depthGT_files)


trans = transforms.Compose([transforms.Resize(size=(384, 1280))])

# output_loc = "/HOMES/yigao/KITTI/vkitti_testset"      # for GDM
# output_loc = "/HOMES/yigao/KITTI/vkitti_testset_test/test"      # for GDM test original
output_loc = "/HOMES/yigao/KITTI/sclaing_factor_dataset"      # for GDM
count = 1

maxDepth = 80


for rgb, depthgt in zip(sorted_rgb_files, sorted_depthGT_files):
    # print(rgb)
    # print(depthgt)
    rgb_img = Image.open( test_rgb_path + rgb)#.convert('RGB')           #an additional alpha channel per pixel https://stackoverflow.com/questions/58496858/pytorch-runtimeerror-the-size-of-tensor-a-4-must-match-the-size-of-tensor-b
    depth_img = Image.open(test_depthGT_path + depthgt)
    # print(rgb_img)
    rgb_img = trans(rgb_img)
    depth_img = trans(depth_img)

    rgb_img = np.asarray(rgb_img)
    depth_img = np.asarray(depth_img)
    # print(rgb_img.shape)
    # print(depth_img)
    # print(depth_img)
    # if not os.path.exists(output_loc + "/%#05d" % (count)):
    #     os.mkdir(output_loc + "/%#05d" % (count))
    # np.save(output_loc + "/%#05d" % (count) + "/image" + "%#05d" % (count), rgb_img)
    # np.save(output_loc + "/%#05d" % (count) + "/depth" + "%#05d" % (count), depth_img)

    # depth_img = cv2.normalize(depth_img, depth_img, 0.8, 80, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_img = depth_img / 100
    print(depth_img)
    np.savez_compressed(output_loc + "/%#05d" % (count), image=rgb_img, depth=depth_img)

    count += 1
