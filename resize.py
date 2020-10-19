from PIL import Image
import json
import numpy as np
import cv2
import math
from PIL import Image


def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return r, g, b


def change_all_pixels(input_pixels, w, h):
    r = []
    g = []
    b = []
    for i in range(0, len(input_pixels)):
        temp_r = []
        temp_g = []
        temp_b = []
        for j in range(0, len(input_pixels[0])):
            r_pixel, g_pixel, b_pixel = Hex_to_RGB(input_pixels[i][j])
            temp_r.append(r_pixel)
            temp_g.append(g_pixel)
            temp_b.append(b_pixel)
        r.append(temp_r)
        b.append(temp_b)
        g.append(temp_g)
    np_r = np.array(r, dtype=np.uint8)
    np_g = np.array(g, dtype=np.uint8)
    np_b = np.array(b, dtype=np.uint8)
    return [np_r, np_g, np_b]


def merge_pic(r, g, b):
    np_r = np.array(r, dtype=np.uint8)
    np_g = np.array(g, dtype=np.uint8)
    np_b = np.array(b, dtype=np.uint8)
    img = cv2.merge([np_b, np_g, np_r])
    return img


def find_best_ratio(target_ratio, w, h):
    best_multiple_w = 0
    best_multiple_h = 0
    min_loss = 100000
    min_m = max(1, int(6000 / w))
    min_n = max(1, int(6000 / h))
    max_m = min(min_m + 500, int(15000 / w) + 1)
    max_n = min(min_n + 500, int(15000 / h) + 1)
    for m in range(min_m, max_m):
        for n in range(min_n, max_n):
            if abs((m / n) - target_ratio) < min_loss:
                best_multiple_w = m
                best_multiple_h = n
                min_loss = abs((m / n) - target_ratio)
    return best_multiple_w, best_multiple_h


def cal_size(h, w, target_h, target_w):
    h1 = int(h * target_h / target_w)
    w1 = int(w * target_w / target_h)
    ceil_after_resize_h = 0
    ceil_after_resize_w = 0
    print("w1: {}, h1: {}".format(w1, h1))
    if abs(h1 - h) < abs(w1 - w) and abs(h1 - h) < 5:
        ceil_after_resize_w = w
        ceil_after_resize_h = h1
        print("img_w: {}, img_h: {}, dst_w: {}, dst_h: {}".format(w, h, w, h1))
    elif abs(h1 - h) > abs(w1 - w) and abs(w1 - w) < 5:
        ceil_after_resize_w = w1
        ceil_after_resize_h = h
        print("img_w: {}, img_h: {}, dst_w: {}, dst_h: {}".format(w, h, w1, h))
    elif h - h1 > 0:
        ceil_after_resize_w = w
        ceil_after_resize_h = h1
        print("img_w: {}, img_h: {}, dst_w: {}, dst_h: {}".format(w, h, w, h1))
    elif w - w1 > 0:
        ceil_after_resize_w = w1
        ceil_after_resize_h = h
        print("img_w: {}, img_h: {}, dst_w: {}, dst_h: {}".format(w, h, w1, h))
    else:
        ceil_after_resize_w = w
        ceil_after_resize_h = h
        print("img_w: {}, img_h: {}, dst_w: {}, dst_h: {}".format(w, h, w, h))
    print("resize_ceil_w: {}, resize_cell_h: {}".format(ceil_after_resize_w, ceil_after_resize_h))
    return ceil_after_resize_w, ceil_after_resize_h


def create_big_img(img_cells, ceil_after_resize_w, ceil_after_resize_h, imgID):
    h, w = img_cells[0].shape
    best_multiple_w = int(1000 / w)
    best_multiple_h = int(1000 / h)
    cell_img = cv2.merge([img_cells[2], img_cells[1], img_cells[0]])
    cv2.imwrite("res/before.jpg", cell_img)
    tile_imgs = []
    for i in range(3):
        tile_imgs.append(np.tile(img_cells[i], (best_multiple_h, best_multiple_w)))
    dst = cv2.merge([tile_imgs[2], tile_imgs[1], tile_imgs[0]])
    return dst, best_multiple_w, best_multiple_h


def get_all_rgb(origin_img):
    all_colors = set()
    h, w, _ = origin_img.shape
    for i in range(0, h):
        for j in range(0, w):
            color = "|".join([str(x) for x in origin_img[i][j]])
            all_colors.add(color)
    return all_colors


def correction_color(color, all_rgbs):
    rgb = [int(i) for i in color.split("|")]
    min_distance = 10000000
    correct_color = rgb
    for c in all_rgbs:
        origin_rgb = [int(i) for i in c.split("|")]
        vec1 = np.array(origin_rgb)
        vec2 = np.array(rgb)
        distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
        if distance < min_distance:
            min_distance = distance
            correct_color = origin_rgb
    return correct_color

def cut_cell_img(img, i, j, N=8):
    h, w, _ = img.shape
    x1 = 0 if i-N < 0 else i-N
    x2 = h if i+N > h else i+N
    y1 = 0 if j-N > 0 else j-N
    y2 = w if j+N < w else j+N
    return x1, x2, y1, y2


def re_color(origin_img, img):
    h, w, _ = img.shape
    print(h, w)
    for i in range(0, h):
        for j in range(0, w):
            color = "|".join([str(x) for x in img[i][j]])
            x1, x2, y1, y2 = cut_cell_img(origin_img, i, j)
            all_rgbs = get_all_rgb(origin_img[x1: x2, y1: y2])
            if color not in all_rgbs:
                correct_color = correction_color(color, all_rgbs)
                img[i][j][0] = correct_color[0]
                img[i][j][1] = correct_color[1]
                img[i][j][2] = correct_color[2]
    return img


def resize_img(img, best_multiple_w, best_multiple_h, ceil_after_resize_w, ceil_after_resize_h, imgID):
    h, w, _ = img.shape
    img_after_resize_w = best_multiple_w * ceil_after_resize_w
    img_after_resize_h = best_multiple_h * ceil_after_resize_h
    print("img_w:{}, img_h:{}, dst_w:{}, dst_h:{}".format(w, h, img_after_resize_w, img_after_resize_h))
    dst = cv2.resize(img, (img_after_resize_w, img_after_resize_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite("res/bigimage.jpg", dst)
    hsv_img = dst[0: ceil_after_resize_h, 0: ceil_after_resize_w]
    #_ = re_color(img, hsv_img)
    cv2.imwrite("res/result_{}.jpg".format(imgID), hsv_img)
    return "res/result_{}.jpg".format(imgID)

def pil_resize(img, best_multiple_w, best_multiple_h, ceil_after_resize_w, ceil_after_resize_h, imgID):
    h, w, _ = img.shape
    img_after_resize_w = best_multiple_w * ceil_after_resize_w
    img_after_resize_h = best_multiple_h * ceil_after_resize_h
    print("img_w:{}, img_h:{}, dst_w:{}, dst_h:{}".format(w, h, img_after_resize_w, img_after_resize_h))
    before_big_img = cv2.imwrite("res/before_big.jpg", img)
    im_pil = Image.open("res/before_big.jpg")
    dst = im_pil.resize((img_after_resize_w, img_after_resize_h), Image.ANTIALIAS)
    dst.save("res/bigimage.jpg")
    big_img = cv2.cvtColor(np.asarray(dst), cv2.COLOR_RGB2BGR)
    hsv_img = big_img[0: ceil_after_resize_h, 0: ceil_after_resize_w]
    cv2.imwrite("res/result_{}.jpg".format(imgID), hsv_img)
    return "res/result_{}.jpg".format(imgID)

def process_image(data):
    imgID = data["imgID"]
    h = len(data["pailie"])
    w = len(data["pailie"][0])
    jingwei_data = data["jingwei"].split("*")
    target_h = int(jingwei_data[0])
    target_w = int(jingwei_data[1])
    img_cells = change_all_pixels(data["pailie"], w, h)
    print("target_w:{}, target_h:{}".format(target_w, target_h))
    ceil_after_resize_w, ceil_after_resize_h = cal_size(h, w, target_h, target_w)
    big_img, best_multiple_w, best_multiple_h = create_big_img(img_cells, ceil_after_resize_w,
                                                                        ceil_after_resize_h, imgID)
    image_path = resize_img(big_img, best_multiple_w, best_multiple_h, ceil_after_resize_w,
                           ceil_after_resize_h, imgID)
    return image_path
    # img_res, best_multiple_w, best_multiple_h = resize_pic(img_cells, target_h, target_w)
    # image_path = re_optimizing(img_res, target_h, target_w, best_multiple_w, best_multiple_h)
    # return image_path


# def local_process_image():
#     img = cv2.imread("img/origin_1.png")
#     all_rgbs = get_all_rgb(img)
#     resize_img = cv2.resize(img, (680, 480), interpolation=cv2.INTER_AREA)
#     #_ = re_color(all_rgbs, resize_img)
#
# local_process_image()
