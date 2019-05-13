#!/usr/bin/env python
import random
import socket
import time

import freenect
import cv2
import numpy as np
import torch
from skimage.feature import canny
from skimage.filters import gaussian

from models import Darknet, load_darknet_weights, non_max_suppression
from settings import RISKY_CLASSES, CAMERA_FOV
from utils.datasets import letterbox
from utils.utils import load_classes, scale_coords, plot_one_box

cv2.namedWindow('LaneWarn')


def get_depth():
    # Get raw depth
    raw = np.array(freenect.sync_get_depth()[0])

    # Magic
    np.clip(raw, 0, 2 ** 10 - 1, raw)
    raw >>= 2
    raw = raw.astype(np.uint8)
    return raw


def get_rgb():
    raw = np.array(freenect.sync_get_video()[0])
    return raw[:, :, ::-1]


def get_bgr():
    return np.array(freenect.sync_get_video()[0])


############
############ DEPTH DATA ANALYSIS
############

def analyse_depth():
    depth = get_depth()
    depth_swap = np.swapaxes(depth, 0, 1)

    depthStrip1d = np.array([np.min(np.sort(stripe)[15:]) for stripe in depth_swap]).astype(np.uint8)
    depthStrip2d_swap = np.array([np.ones(depth_swap.shape[1]) * depth for depth in depthStrip1d]).astype(np.uint8)
    depthStrip2d_swap_rgb = np.stack((depthStrip2d_swap,) * 3, axis=-1)

    deapthEdge1d = np.zeros(depthStrip1d.shape)

    # state False=white True=Black
    state = False
    for i, _ in np.ndenumerate(deapthEdge1d[:-1]):
        i = i[0]
        if state == False:
            if depthStrip1d[i] < 230:
                state = True
        if state == True:
            if depthStrip1d[i] > 240:
                state = False
        deapthEdge1d[i] = not state

    # Remove tiny black bars
    blackcounter = 0
    for i, _ in np.ndenumerate(deapthEdge1d[:-1]):
        i = i[0]
        if deapthEdge1d[i] == True:
            if blackcounter > 0 and blackcounter < 15:
                for i in range(max(0, i - 15), i):
                    deapthEdge1d[i] = True
            blackcounter = 0
        if deapthEdge1d[i] == False:
            blackcounter += 1

    # Remove tiny black bars
    whitecounter = 0
    for i, _ in np.ndenumerate(deapthEdge1d[:-1]):
        i = i[0]
        if deapthEdge1d[i] == False:
            if whitecounter > 0 and whitecounter < 10:
                for i in range(max(0, i - 10), i):
                    deapthEdge1d[i] = False
            whitecounter = 0
        if deapthEdge1d[i] == True:
            whitecounter += 1

    objects = ""
    lastStart = 0
    for i, _ in np.ndenumerate(deapthEdge1d[:-2]):
        i = i[0]
        if deapthEdge1d[i] == False:
            depthStrip2d_swap_rgb[i][-0] = [0, 255, 255]
            depthStrip2d_swap_rgb[i][-1] = [0, 255, 255]
            depthStrip2d_swap_rgb[i][-2] = [0, 255, 255]
            depthStrip2d_swap_rgb[i][-3] = [0, 255, 255]
            depthStrip2d_swap_rgb[i][-4] = [0, 255, 255]
            depthStrip2d_swap_rgb[i][-5] = [0, 255, 255]

        # End of zone
        if deapthEdge1d[i] == False and deapthEdge1d[i + 1] == True:
            depthStrip2d_swap_rgb[i] = [[0, 255, 0] for pixel in depthStrip2d_swap_rgb[i]]
            depthStrip2d_swap_rgb[i + 1] = [[0, 255, 0] for pixel in depthStrip2d_swap_rgb[i]]

            # Average object distance
            avrg = (np.average(depthStrip1d[lastStart:i]) * (-1) + 255) / 255
            mini = (np.min(depthStrip1d[lastStart:i])) / 255 * 3.141569  # Unit transformation to m

            # Append object
            angle = (i / len(deapthEdge1d) * 2 - 1) * 33  # 33=fov
            objects += "obstacle;" + str(angle) + ";" + str(mini) + "|"

            # Center indicator with variable height
            for y in range(0, depthStrip2d_swap_rgb.shape[1]):
                if ((y / depthStrip2d_swap_rgb.shape[1]) > avrg):
                    depthStrip2d_swap_rgb[int(lastStart + ((i - lastStart) / 2)) + 0][y] = [255, 0, 255]
                    depthStrip2d_swap_rgb[int(lastStart + ((i - lastStart) / 2)) - 1][y] = [255, 0, 255]
                    depthStrip2d_swap_rgb[int(lastStart + ((i - lastStart) / 2)) + 1][y] = [255, 0, 255]

        # Start of zone
        if deapthEdge1d[i] == True and deapthEdge1d[i + 1] == False:
            depthStrip2d_swap_rgb[i] = [[0, 0, 255] for pixel in depthStrip2d_swap_rgb[i]]
            depthStrip2d_swap_rgb[i + 1] = [[0, 0, 255] for pixel in depthStrip2d_swap_rgb[i]]

            lastStart = i

    # Remove last bash from objects string
    if len(objects) > 0:
        objects = objects[:-1]

    # Rotate image matrix
    depthStrip2d_rgb = np.swapaxes(depthStrip2d_swap_rgb, 0, 1)

    # Give depth 3rd dimension
    depth = np.stack((depth,) * 3, axis=-1)

    return {"raw": depth, "done": depthStrip2d_rgb, "objects": objects}


def analyse_rgb():
    t = time.time()
    img0 = get_bgr()
    img_org = img0[:, :, ::-1]
    img, _, _, _ = letterbox(img0, new_shape=image_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    img = torch.from_numpy(img).unsqueeze(0).to('cpu')
    pred, _ = model(img)

    det = non_max_suppression(pred, 0.6, 0.5)[0]

    if det is not None and len(det) > 0:
        detected_classes = []
        print('+ Rescaling model')
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *coordinates, conf, cls_conf, cls in det:
            if classes[int(cls)] in RISKY_CLASSES:
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(coordinates, img0, label=label, color=colors[int(cls)])
                print(f"+ Detected {classes[int(cls)]}")
                detected_classes.append({classes[int(cls)]: {'x': coordinates[0], 'y': coordinates[1]}})

        n = []
        for counter in detected_classes:
            width = img0.shape[1]
            x, y = counter[list(counter.keys())[0]].values()
            phi = (x / width * 2 - 1) * (CAMERA_FOV / 2)
            n.append(f"{list(counter.keys())[0]};{phi};-1|")

        s = str(''.join(str(x) for x in n)[:-1])

        return {"raw": get_rgb(), "done": img0, "objects": s}
    return {'raw': img_org, 'done': img_org, 'objects': ''}


############
############ MAIN LOOP
############
def app():
    cfg = 'ml-data/yolov3.cfg'
    global image_size
    image_size = 320
    weights = 'ml-data/weights/yolov3.weights'
    classes_file = 'ml-data/classes.txt'
    socket_ip = '10.10.10.1'
    # socket_ip = '127.0.0.1'
    socket_port = 1337

    print('+ Initializing model')
    global model
    model = Darknet(cfg, image_size)
    print('+ Loading model')
    load_darknet_weights(model, weights)
    print('+ Fusing model')
    model.fuse()
    print('+ Loading model to CPU')
    model.to('cpu').eval()
    print('+ Loading classes')
    global classes
    classes = load_classes(classes_file)
    global colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    print('+ Connecting to remote socket')
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((socket_ip, socket_port))

    while 1:
        #
        # Depth
        #
        depth_result = analyse_depth()
        depth_raw = depth_result["raw"]
        depth_done = depth_result["done"]
        depth_objects = depth_result["objects"]

        #
        # RGB
        #
        rgb_result = analyse_rgb()
        rgb_raw = rgb_result["raw"]
        rgb_done = rgb_result["done"]
        rgb_objects = rgb_result["objects"]

        print("FRAME [D]: " + depth_objects)
        print("FRAME [C]: " + rgb_objects)

        sock.send(bytes(f"{rgb_objects}|{depth_objects}".encode('utf-8')))

        time.sleep(0.01)

        # Plot
        vbar = np.zeros((depth_raw.shape[0], 5, 3)).astype(np.uint8)
        depthbar = np.concatenate((depth_raw, vbar, depth_done), axis=1)
        rgbbar = np.concatenate((rgb_raw, vbar, rgb_done), axis=1)
        hbar = np.zeros((5, depthbar.shape[1], 3)).astype(np.uint8)
        cv2.imshow('LineWarn', np.concatenate((depthbar, hbar, rgbbar), axis=0))

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__': app()
