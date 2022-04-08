from os.path import join
from os import listdir
import cv2
import json
import sys

sys.path.append('/p300/ihuman/code/ins_label/integrate')

from schedule import Schedule

"""
对原图进行crop操作，输入有原始图片与标注图片
输出cropped图片与crop information
"""

# 参数
output_crop_img = True  # 是否保存crop后的img
output_crop_info = True  # 是否保存crop的信息

print(f'output_crop_img: {output_crop_img}')
print(f'output_crop_info: {output_crop_info}')
print('')

# 设定
label_dir = '/group/xiangyi/Chuanyang/labels/train'
raw_imgdir = '/group/xiangyi/Chuanyang/raw_images/train'
crop_imgdir = '/group/xiangyi/Chuanyang/crop_images/train'
cell_pattern = ''
crop_outdir = '/p300/ihuman/dataset/crop_images/train'
anno_outpath = '/p300/ihuman/dataset/annotations/train/crop_info_train.json'


# 获取931_14所有相关数据file path
cells = []
for i in listdir(label_dir):
    if cell_pattern in i:
        cells.append(i)


# 执行逻辑
# 1-1 处理label
info = {
    'info': []
}
schedule = Schedule(cells)
for count, file_name in enumerate(cells, 1):
    now_info = {
        'img_name': file_name
    }
    label_path = join(label_dir, file_name)
    la = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    h, w = la.shape
    now_info['raw_h'] = h
    now_info['raw_w'] = w
    left = sys.maxsize
    right = -1
    for i in range(w):
        temp = la[:, i]
        if any(temp):
            left = i
            break
    for i in reversed(range(w)):
        temp = la[:, i]
        if any(temp):
            right = i+1
            break
    # crop image
    raw_imgpath = join(raw_imgdir, file_name)
    raw_img = cv2.imread(raw_imgpath, cv2.IMREAD_GRAYSCALE)
    raw_img = raw_img[:, left:right]
    now_info['left'] = left
    now_info['right'] = right
    # save cropped image
    crop_imgpath = join(crop_outdir, file_name)
    if output_crop_img:
        cv2.imwrite(crop_imgpath, raw_img)
    info['info'].append(now_info)
    schedule.watch()

if output_crop_info:
    with open(anno_outpath, 'w', encoding='utf-8') as f:
        json.dump(info, f)

