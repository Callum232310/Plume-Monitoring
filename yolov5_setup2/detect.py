# -*- coding: utf-8 -*-
"""
Script Name: detect.py

Description: This script contains the main functions required to make inferences using a trained YOLOv5 model.

Original Author: Glenn Jocher
Modified by: Callum O'Donovan

Original Creation Date: 2020
Modification Date: September 26th 2022

Email: callumodonovan2310@gmail.com

Acknowledgements:
    - Glenn Jocher (Ultralytics, LLC.)

Modifications:
    - This script has been adapted to work for the plume case study.
    
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license



import numpy as np
import pandas as pd
import math
import argparse
import os
import sys
from pathlib import Path
import csv
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, non_max_suppression, scale_coords,
                           check_imshow, xyxy2xywh, increment_path)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    startTime = time.time()
    
    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names


    #Validation data
    realDiameters = [74.2331288343558, 76.6871165644172, 70.5521472392638, 69.6319018404908, 
                     69.0184049079755, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     71.1656441717791, 74.8466257668712, 85.2760736196319, 97.239263803681, 
                     107.361963190184, 110.429447852761, 115.030674846626, 123.006134969325, 
                     125.460122699387, 128.834355828221, 129.447852760736, 131.901840490798, 
                     131.901840490798, 130.981595092025, 125.766871165644, 125.766871165644, 
                     124.539877300613, 122.699386503067, 116.564417177914, 109.815950920245, 
                     101.226993865031, 96.0122699386503, 85.8895705521472, 82.8220858895705, 
                     80.9815950920245, 76.6871165644172, 71.1656441717791, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     72.6993865030675, 76.6871165644172, 87.7300613496932, 92.0245398773006, 
                     99.079754601227, 107.361963190184, 115.950920245399, 121.165644171779, 
                     123.312883435583, 125.766871165644, 127.300613496933, 134.355828220859, 
                     138.036809815951, 139.570552147239, 147.239263803681, 148.466257668712, 
                     146.319018404908, 145.705521472393, 145.092024539877, 144.171779141104, 
                     143.558282208589, 142.944785276074, 142.944785276074, 141.104294478528, 
                     131.288343558282, 122.699386503067, 122.085889570552, 117.484662576687, 
                     116.564417177914, 113.496932515337, 111.963190184049, 104.294478527607, 
                     98.4662576687116, 92.0245398773006, 79.1411042944785, 76.6871165644172, 
                     72.0858895705521, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 71.1656441717791, 73.6196319018405, 
                     78.5276073619632, 79.7546012269939, 76.6871165644172, 70.5521472392638, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     72.0858895705521, 73.0061349693251, 74.8466257668712, 76.6871165644172, 
                     79.7546012269939, 80.6748466257669, 82.2085889570552, 82.8220858895705, 
                     79.7546012269939, 76.0736196319018, 75.1533742331288, 75.1533742331288, 
                     75.1533742331288, 73.6196319018405, 70.5521472392638, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 69.0184049079755, 
                     69.9386503067485, 70.5521472392638, 69.9386503067485, 68.4049079754601, 
                     67.4846625766871, 71.1656441717791, 73.6196319018405, 75.1533742331288, 
                     76.6871165644172, 77.3006134969325, 76.6871165644172, 76.6871165644172, 
                     77.6073619631902, 78.8343558282209, 79.7546012269939, 83.4355828220859, 
                     85.8895705521472, 86.5030674846626, 94.478527607362, 92.0245398773006, 
                     91.4110429447853, 87.4233128834356, 85.2760736196319, 82.8220858895705, 
                     81.9018404907975, 79.7546012269939, 76.0736196319018, 74.2331288343558, 
                     73.0061349693251, 71.7791411042945, 70.5521472392638, 69.6319018404908, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 68.4049079754601, 
                     70.2453987730061, 73.6196319018405, 83.7423312883435, 94.478527607362, 
                     105.828220858896, 113.496932515337, 121.779141104294, 128.834355828221, 
                     141.717791411043, 143.558282208589, 144.171779141104, 145.705521472393, 
                     146.932515337423, 150.306748466258, 150.613496932515, 152.147239263804, 
                     153.374233128834, 151.840490797546, 151.840490797546, 151.533742331288, 
                     151.226993865031, 150.920245398773, 150.306748466258, 144.171779141104, 
                     136.80981595092, 134.969325153374, 134.049079754601, 125.766871165644, 
                     120.245398773006, 113.496932515337, 106.748466257669, 101.840490797546, 
                     95.0920245398773, 92.0245398773006, 83.4355828220859, 70.5521472392638, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     68.0981595092024, 72.0858895705521, 78.8343558282209, 78.8343558282209, 
                     76.6871165644172, 73.0061349693251, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 75.4601226993865, 
                     76.6871165644172, 85.8895705521472, 101.840490797546, 102.760736196319, 
                     107.361963190184, 117.484662576687, 123.006134969325, 131.901840490798, 
                     134.355828220859, 136.196319018405, 138.036809815951, 141.717791411043, 
                     147.239263803681, 148.466257668712, 148.773006134969, 149.079754601227, 
                     149.079754601227, 149.079754601227, 148.773006134969, 148.159509202454, 
                     148.159509202454, 147.239263803681, 147.239263803681, 146.01226993865, 
                     145.398773006135, 137.423312883436, 128.834355828221, 123.926380368098, 
                     121.779141104294, 116.564417177914, 114.110429447853, 107.361963190184, 
                     100.613496932515, 85.8895705521472, 79.7546012269939, 75.1533742331288, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 70.2453987730061, 73.6196319018405, 84.9693251533742, 
                     85.8895705521472, 82.5153374233129, 78.8343558282209, 78.2208588957055, 
                     73.0061349693251, 70.5521472392638, 67.4846625766871, 67.4846625766871, 
                     69.0184049079755, 73.0061349693251, 75.4601226993865, 77.6073619631902, 
                     82.8220858895705, 86.8098159509202, 94.1717791411043, 98.159509202454, 
                     101.226993865031, 102.453987730061, 104.294478527607, 104.907975460123, 
                     106.441717791411, 107.668711656442, 107.361963190184, 107.361963190184, 
                     107.361963190184, 105.828220858896, 105.828220858896, 104.294478527607, 
                     104.294478527607, 98.159509202454, 98.159509202454, 96.9325153374233, 
                     95.0920245398773, 92.0245398773006, 86.5030674846626, 82.8220858895705, 
                     81.2883435582822, 71.7791411042945, 70.5521472392638, 69.3251533742331, 
                     68.7116564417178, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     69.6319018404908, 71.1656441717791, 82.2085889570552, 84.3558282208589, 
                     88.0368098159509, 88.9570552147239, 87.4233128834356, 84.9693251533742, 
                     76.6871165644172, 68.4049079754601, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 69.0184049079755, 
                     70.5521472392638, 72.6993865030675, 79.7546012269939, 87.7300613496932, 
                     91.4110429447853, 98.159509202454, 101.840490797546, 103.374233128834, 
                     104.294478527607, 105.521472392638, 107.361963190184, 107.975460122699, 
                     108.588957055215, 107.361963190184, 107.055214723926, 105.828220858896, 
                     105.828220858896, 98.159509202454, 94.7852760736196, 85.8895705521472, 
                     82.8220858895705, 73.6196319018405, 71.1656441717791, 68.4049079754601, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     68.0981595092024, 69.0184049079755, 70.5521472392638, 70.8588957055215, 
                     70.8588957055215, 69.9386503067485, 69.0184049079755, 68.4049079754601, 
                     67.4846625766871, 68.7116564417178, 70.5521472392638, 70.5521472392638, 
                     69.6319018404908, 68.7116564417178, 69.9386503067485, 70.5521472392638, 
                     71.1656441717791, 72.0858895705521, 76.6871165644172, 76.6871165644172, 
                     78.5276073619632, 78.8343558282209, 79.4478527607362, 79.7546012269939, 
                     79.7546012269939, 82.8220858895705, 85.2760736196319, 85.8895705521472, 
                     85.8895705521472, 87.4233128834356, 87.7300613496932, 87.7300613496932, 
                     88.0368098159509, 87.4233128834356, 85.5828220858896, 83.1288343558282, 
                     82.8220858895705, 76.6871165644172, 76.6871165644172, 73.0061349693251, 
                     68.7116564417178, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 68.7116564417178, 
                     71.4723926380368, 73.6196319018405, 75.7668711656442, 78.2208588957055, 
                     79.4478527607362, 79.1411042944785, 79.1411042944785, 75.4601226993865, 
                     70.5521472392638, 71.4723926380368, 70.5521472392638, 69.0184049079755, 
                     69.3251533742331, 70.5521472392638, 71.1656441717791, 71.7791411042945, 
                     72.0858895705521, 72.0858895705521, 72.6993865030675, 73.0061349693251, 
                     73.6196319018405, 74.2331288343558, 75.1533742331288, 75.4601226993865, 
                     76.6871165644172, 82.8220858895705, 85.2760736196319, 83.4355828220859, 
                     76.6871165644172, 71.4723926380368, 69.0184049079755, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     70.5521472392638, 74.5398773006135, 84.6625766871166, 86.5030674846626, 
                     88.9570552147239, 89.5705521472393, 91.1042944785276, 92.3312883435583, 
                     92.0245398773006, 89.5705521472393, 82.8220858895705, 80.6748466257669, 
                     80.6748466257669, 82.2085889570552, 82.8220858895705, 73.6196319018405, 
                     75.1533742331288, 76.6871165644172, 79.7546012269939, 80.6748466257669, 
                     82.8220858895705, 81.9018404907975, 80.3680981595092, 78.5276073619632, 
                     78.2208588957055, 77.6073619631902, 77.3006134969325, 77.3006134969325, 
                     76.9938650306748, 76.6871165644172, 68.0981595092024, 68.0981595092024, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 67.4846625766871, 67.4846625766871, 67.4846625766871, 
                     67.4846625766871, 68.4049079754601, 69.6319018404908, 70.8588957055215, 
                     80.9815950920245, 88.9570552147239, 91.1042944785276, 100, 
                     102.760736196319, 114.110429447853, 116.564417177914, 122.699386503067, 
                     126.687116564417, 128.834355828221, 131.901840490798, 137.423312883436, 
                     138.650306748466, 139.570552147239, 141.717791411043, 144.171779141104, 
                     145.705521472393, 147.239263803681, 146.625766871166, 141.104294478528, 
                     138.036809815951, 134.969325153374, 133.128834355828]

    #Initialise
    IDlist = []
    IDlist2 = []

    finalResults = []
    
    tuyerePlumeWidthCounter = 0
    zeroPlumeWidthCounter = 0
    
    savedContactTime = 0
    savedContactWidth = 0
    
    IDcounter = 0
    contactCount = 0
    nonZeroCounter = 0
    detCounter = 0
    
    numberofDets = 0

    fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    
    tuyerePlumeWidthTable = pd.DataFrame(columns=["frameNumber", "tuyerePlumeWidth (mm)"])
    tuyerePlumeWidthTable2Header = ["frameNumber", "tuyerePlumeWidth (mm)", "actualPlumeWidth (mm)", "error %"]
    tuyerePlumeWidthTable2 = []
    
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        tuyereContactSwitch = 0
        zeroPlumeWidthFrameSwitch = 0
        annotationMask = np.zeros((768,768))
        
        cv2.imwrite("annotationMask.png", annotationMask) # imwrite(filename, img[, params])
        
        annotationMask = cv2.imread("annotationMask.png")
        
        frameContactCheck = 0
        frameClockDet = 1
        
        #cv2.imwrite('image1.jpg', img)

        img2 = img.copy()

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        
        #cv2.imwrite('image2.jpg', img)
        
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        #cv2.imwrite('image3.jpg', img)
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        t2 = time_sync()
        dt[0] += t2 - t1
        
        #cv2.imwrite('image4.jpg', img2)

        img2 = np.moveaxis(img2, -1, 0)
        img2 = np.moveaxis(img2, -1, 0)    
        img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img3 = cv2.convertScaleAbs(img3, alpha=1.95, beta=50)

        bgsegMask = fgbg.apply(img3)
                
        kernel = np.ones((5,5), np.uint8)
        bgsegMask = cv2.erode(bgsegMask, kernel) 
        bgsegMask = cv2.dilate(bgsegMask, kernel)
        
        fgmaskRGB = cv2.cvtColor(bgsegMask,cv2.COLOR_GRAY2RGB)
        fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 255)] = (255,0,255)

        src1 = fgmaskRGB
        src2 = img2
        dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
        
        lightPink = (200, 85, 190)
        darkPink = (255, 149, 255)
        mask = cv2.inRange(dst, lightPink, darkPink)
        fgmaskRGB[np.where(mask !=0)] = (0,0,0)
        bgsegMask[np.where(mask !=0)] = 0

        tuyereRegionMaskLeft = bgsegMask[700:730, 225:325]
        tuyereRegionMaskRight = bgsegMask[700:730, 450:550]

        edgesLeft = cv2.Canny(tuyereRegionMaskLeft,1,1,apertureSize = 7)
        edgesRight = cv2.Canny(tuyereRegionMaskRight,1,1,apertureSize = 7)

        lineFrame = np.zeros((768,768), np.uint8)
        try:
            linesLeft = cv2.HoughLinesP(edgesLeft, rho=1, theta=np.pi/180, threshold=20, minLineLength=1, maxLineGap=12)
            linesRight = cv2.HoughLinesP(edgesRight, rho=1, theta=np.pi/180, threshold=20, minLineLength=1, maxLineGap=12)
            
            for lineLeft in linesLeft:
                lx1,ly1,lx2,ly2 = lineLeft[0]
                dY = ly2 - ly1
                dX = lx2 - lx1
                angleInDegrees = math.atan2(dY, dX) * 180 / math.pi
                
                if  angleInDegrees < 0:
                    cv2.line(img3[700:730, 225:325],(lx1,ly1),(lx2,ly2),(0,255,0),2)    
                    cv2.line(lineFrame[700:730, 225:325],(lx1,ly1),(lx2,ly2),(255,255,255),1)    

            for lineRight in linesRight:
                lx1,ly1,lx2,ly2 = lineRight[0]
                dY = ly2 - ly1
                dX = lx2 - lx1
                angleInDegrees = math.atan2(dY, dX) * 180 / math.pi
                
                if 0 < angleInDegrees:
                    cv2.line(img3[700:730, 450:550],(lx1,ly1),(lx2,ly2),(0,255,0),2)
                    cv2.line(lineFrame[700:730, 450:550],(lx1,ly1),(lx2,ly2),(255,255,255),1)
          
        except:
            pass
        
        lineKernel = np.ones((10, 10), np.uint8)
        lineFrame = cv2.dilate(lineFrame, lineKernel) 
        
        fgmaskRGB[np.where(lineFrame == 255)] = (0,0,0)
        bgsegMask[np.where(lineFrame == 255)] = 0
               
        dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        #cv2.imwrite('image5.jpg', bgsegMask)

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        
                        detCounter +=1
                        numberofDets +=1

                        if bboxes[1] < 710 and bboxes[3] > 710: #if bounding box crosses over measurement line

                            bboxSegTest = bgsegMask[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
                            bboxLen = bboxes[3] - bboxes[1]
                            bboxImgDiff = bboxes[3] - 710
                            bboxImgComp = bboxLen - bboxImgDiff
                            
                            bboxSegIndices = np.argwhere(bboxSegTest == 255) #indices of white pixels within bbox                     
                            bboxSegIndicesAtLine = np.argwhere(bboxSegIndices[:,0] == bboxImgComp) 
                            
                            if any(bboxSegIndices[bboxSegIndicesAtLine,1]) == True:
                                
                                minWidthAtLine = np.min(bboxSegIndices[bboxSegIndicesAtLine,1]) #IN PIXELS
                                maxWidthAtLine = np.max(bboxSegIndices[bboxSegIndicesAtLine,1])

                                tuyerePlumeWidth = maxWidthAtLine - minWidthAtLine #IN PIXELS
                                tuyerePlumeWidthCounter = 1
                                tuyerePlumeWidthMM = "{:.2f}".format(tuyerePlumeWidth/1.84)
                                
                                tuyerePlumeWidthTable.loc[frame_idx,'frameNumber'] = frame_idx
                                tuyerePlumeWidthTable.loc[frame_idx,'tuyerePlumeWidth (mm)'] = tuyerePlumeWidthMM
                        
                                nonZeroCounter +=1
                                savedTuyerePlumeWidth = tuyerePlumeWidthMM

                                zeroPlumeWidthFrameSwitch = 2 #2 indicates a relevant box has been found in this frame

                        else: 
                            
                            if zeroPlumeWidthFrameSwitch != 2: #if a relevant box has not been found so far
                                zeroPlumeWidthFrameSwitch = 1
                                tuyerePlumeWidth = 0
                                tuyerePlumeWidthMM = 0
                                tuyerePlumeWidthCounter = 0
                                
                        #Ensure IDs are consecutive
                        maskID = id
                        
                        if id not in IDlist:
                          IDcounter +=1
                          maskID = IDcounter
                          IDlist2.append(maskID)
                          IDlist.append(id)

                        elif id in IDlist:
                          maskID = IDlist2[IDlist.index(id)]
                          IDlist.append(id)
                          IDlist2.append(maskID)
                          
                        #Calculate height and width in MM
                        x1 = bboxes[0]
                        y1 = bboxes[1]
                        x2 = bboxes[2]
                        y2 = bboxes[3]
                        
                        height = y2-y1
                        width = x2-x1
                        
                        heightMM = "{:.2f}".format(height/1.84)
                        widthMM = "{:.2f}".format(width/1.84)
                        
                        if cls == 2:
                          clsName = 'jetting'
                          jettingHeight = heightMM
                          jettingWidth = widthMM
                          
                        elif cls == 1:
                          clsName = 'forming'
                          jettingHeight = "{:.2f}".format(0)
                          jettingWidth = "{:.2f}".format(0)
                          
                        elif cls == 0:
                          clsName = 'collapsing/collapsed'
                          jettingHeight = "{:.2f}".format(0)
                          jettingWidth = "{:.2f}".format(0)
                          
                        timedInf = time.time() - startTime
                        timedInf = "{:.2f}".format(timedInf)
      
                        if tuyerePlumeWidthMM != "{:.2f}".format(0) and tuyerePlumeWidthMM != 0:
                            frameContactCheck = 1                 
                            plumeContactTime = time.time() - startTime     
                            actualContactTime = plumeContactTime - savedContactTime   
                            savedContactTime = plumeContactTime
                            
                            #if tuyere contact occurred during this detection, check if previous detection had contact
                            if savedContactWidth == 0:
                                contactCount +=1
                                savedContactWidth = tuyerePlumeWidthMM
                                
                        else:
                            plumeContactTime = 0
                            actualContactTime = 0

                        plumeContactTime = "{:.2f}".format(plumeContactTime)  
    
                        #Output results
                        finalResult = [time.time()-startTime, str(maskID), clsName, heightMM, widthMM, jettingHeight, jettingWidth, tuyerePlumeWidthMM, plumeContactTime, actualContactTime, "{:.2f}".format(contactCount), timedInf]
                        
                        print(finalResult)
                        
                        if finalResults == []:
                          finalResults = [finalResult]
                        else:
                          finalResults.append(finalResult)
                          
                        #Save results
                        with open("finalResult%d.csv" %maskID, "a", newline="") as f:
                          writer = csv.writer(f)
                          writer.writerows([finalResult])

                        c = int(cls)  # integer class
                        label = ''
                        annotator.box_label(bboxes, label, color=colors(c, True))


                        cv2.putText(annotationMask, 'Frame: %d' %float(frame_idx), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(annotationMask, 'ID: %d' %float(maskID), (x1+1,y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(annotationMask, 'State: %s' %str(names[c]), (x1+1,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(annotationMask, 'Conf: %s' %"{:.2f}".format(conf.item()), (x1+1,y1+70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(annotationMask, 'Height: %dmm' %float(heightMM), (x1+1,y1+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(annotationMask, 'Width: %dmm' %float(widthMM), (x1+1,y1+110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                        
                        if frameClockDet == 1:
                            cv2.putText(annotationMask, 'Time: %ds' %float(time.time()-startTime), (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.putText(annotationMask, 'Count: %d' %float(contactCount), (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            
                            frameClockDet = 0
                            
                        if tuyerePlumeWidthCounter == 1 and tuyereContactSwitch == 0:
                            
                          cv2.putText(annotationMask, 'TWidth: %dmm' %float(tuyerePlumeWidth), (5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                          
                          cv2.line(annotationMask, (minWidthAtLine+bboxes[0], 710), (maxWidthAtLine+bboxes[0], 710), (0,0,255), 3) #green line
                          cv2.line(annotationMask, (minWidthAtLine+bboxes[0], 705), (minWidthAtLine+bboxes[0], 715), (0,0,255), 3) #green line
                          cv2.line(annotationMask, (maxWidthAtLine+bboxes[0], 705), (maxWidthAtLine+bboxes[0], 715), (0,0,255), 3) #green line
                          
                          tuyereContactSwitch = 1
                          tuyerePlumeWidthCounter = 0

                        if clsName == 'jetting':
                            
                            midpoint = ((x2+x1)/2)
                            
                            cv2.line(annotationMask, (int(midpoint), y1),(int(midpoint),y2), (0,0,255), 2)
                            cv2.line(annotationMask, (int(midpoint)-5, y1),(int(midpoint)+5,y1), (0,0,255), 2)
                            cv2.line(annotationMask, (int(midpoint)-5, y2),(int(midpoint)+5,y2), (0,0,255), 2)
                            
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()

            #Superimpose coloured mask
            fgmaskRGB = cv2.cvtColor(bgsegMask,cv2.COLOR_GRAY2RGB)
            fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 255)] = (255,0,255)
            src1 = fgmaskRGB
            src2 = im0
            dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
            dst = cv2.addWeighted(dst, 1, annotationMask,1 , 0)

            if show_vid:
                cv2.imshow(str(p), im0)
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                        
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer.write(dst)
                
        if frameContactCheck == 0:
            savedContactWidth = 0
            
        if zeroPlumeWidthFrameSwitch == 1 or zeroPlumeWidthFrameSwitch == 0:
            
            try:
              tuyerePlumeWidthTable2.append([frame_idx, 0, realDiameters[frame_idx], 100])
              zeroPlumeWidthCounter += 1

            except:
                  tuyerePlumeWidthTable2Header = ["Frame", "predictedContactWidth(mm)", "measuredContactWidth(mm)", "Error(%)"]
                  
                  with open("tuyerePlumeWidthTable2.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows([tuyerePlumeWidthTable2Header])
                    writer.writerows(tuyerePlumeWidthTable2)
                    
                    break
                
        elif zeroPlumeWidthFrameSwitch == 2:

            try:
              errorPercent = abs((float(realDiameters[int(frame_idx)])-float(savedTuyerePlumeWidth))/float(realDiameters[int(frame_idx)]))*100
              tuyerePlumeWidthTable2.append([frame_idx, savedTuyerePlumeWidth, float(realDiameters[frame_idx]), errorPercent])
              
            except:
                  tuyerePlumeWidthTable2Header = ["Frame", "predictedContactWidth(mm)", "measuredContactWidth(mm)", "Error(%)"]
                  
                  with open("tuyerePlumeWidthTable2.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(tuyerePlumeWidthTable2Header)
                    writer.writerows(tuyerePlumeWidthTable2)
                    
                    break
                
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)

    csvHeader = ["Timestamp", "plumeID", "Class", "Height(mm)", "Width(mm)", "jettingHeight(mm)", "jettingWidth(mm)", "contactWidth(mm)", "contactTime(s)", "actualContactTime(s)", "contactCount", "Time(s)"]
    
    with open("finalResultsArranged.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows([csvHeader])

    for csvFile in range(1,max(IDlist2)+1):   
      with open('finalResult%d.csv' %csvFile, 'r') as f1:
        individual = f1.read()

        with open('finalResultsArranged.csv', 'a') as f2:
            f2.write(individual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
