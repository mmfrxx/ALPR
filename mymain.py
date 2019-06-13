#Parameters to tune:
#	1) Frame steps
#	2) Frame dimensions
#   3) Turn on or off the KZ standards (depending on the goal)
import sys, os
import cv2
import keras
import numpy as np
import traceback
import time
import shutil

import darknet.python.darknet as dn

from src.label 					import Label, lwrite, Shape, writeShapes, dknet_label_conversion, lread, readShapes
from os.path 					import splitext, basename, isdir, isfile
from os 						import makedirs
from src.utils 					import crop_region, image_files_from_folder, im2single, nms
from darknet.python.darknet 	import detect
from glob 						import glob
from src.keras_utils 			import load_model, detect_lp
from src.drawing_utils			import draw_label, draw_losangle, write2img

from pdb import set_trace as pause



vehicle_threshold = .5

vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
vehicle_netcfg  = b'data/vehicle-detector/yolo-voc.cfg'
vehicle_dataset = b'data/vehicle-detector/voc.data'

vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)

lp_threshold = .5

wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
wpod_net = load_model(wpod_net_path)

def adjust_pts(pts,lroi):
    return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

"""with open('./config.json') as f:
    config = json.load(f)

ip = str(config["ip"])"""

def detect_vehicle(frame, framenumber):

    input_dir = b'cuttedframe.jpg'
    output_dir = 'tmp/output'

    if not isdir(output_dir): makedirs(output_dir)

    print("Searching for vehicles using YOLO...")

    print("Scanning frame #{}".format(framenumber))

    R, wh = detect(vehicle_net, vehicle_meta, input_dir, thresh = vehicle_threshold)

    return R

def cutvehicle(frame, n, R, boxforcars):
    print('\t\t%d cars found' % len(R))
    Iorig = frame
    WH = np.array(Iorig.shape[1::-1],dtype=float)
    Lcars = []
    output_dir = 'tmp/output'


    for i,r in enumerate(R):
        x1 = int(r[2][0])-int(r[2][2]/2)
        y1 = int(r[2][1])-int(r[2][3]/2)+280
        x2 = int(r[2][0])+int(r[2][2]/2)
        y2 = int(r[2][1])+int(r[2][3]/2)+280
        
        cv2.imwrite('%s/%s_%dcar.png' % (output_dir,n,i), frame[y1:y2, x1:x2])
        green = (50, 205, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        boxforcars.append(tuple([x1, y1, int(r[2][2]), int(r[2][3])]))
    
def checkfortrack(boxforcars, R):
    for j in boxforcars:
        for i, r in enumerate(R):
            x1 = int(r[2][0])-int(r[2][2]/2)
            y1 = int(r[2][1])-int(r[2][2]/2)+280
            x2 = int(r[2][0])+int(r[2][2]/2)
            y2 = int(r[2][1])+int(r[2][2]/2)+280

            print(j)
        

def crossinglines(frame):
    blue = (255, 0, 0)
    green = (50, 205, 50)
    cv2.line(frame, (0, 280), (1800, 280), blue, 5) # draw first line 
    cv2.line(frame, (0, 580), (1800, 580), green, 5) # draw second line
    return frame


if __name__ == "__main__":
    
    cap = cv2.VideoCapture("atbasar.avi")
    numbFrames = -1
    boxforcars = []

    tracker_type = 'MOSSE'
    trackers = cv2.MultiTracker_create()


    writer = None
    while True:

        numbFrames += 1

        ret, frame = cap.read()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if numbFrames%7==0: print("[INFO] {} total frames in video".format(total))

        #cuttedframe = frame[:] #Parameter to tune

        if ret == False:
            break

        print("Frame number #" + str(numbFrames))
        print(boxforcars)
        if numbFrames % 25 == 0: trackers= cv2.MultiTracker_create()
        if numbFrames % 7==0:
            status = "Detecting"
# #################################   VEHICLE DETECTION   #################################
            start_time = time.time()
            cv2.imwrite('frame.jpg', frame)
            frame_forCarDetection = frame[400:680, 0:1800]
            frame_forCarTracking = frame[400:,:]
            cv2.imwrite('cuttedframe.jpg', frame_forCarDetection)
            R = detect_vehicle(frame_forCarDetection, numbFrames)
            R = [r for r in R if r[0] in [b'car', b'bus']]    
            if len(R) != 0: 
                cutvehicle(frame, numbFrames, R, boxforcars)
                end_time = time.time()
                print("Vehicle detection time: " + str(end_time - start_time))
                trackers.add(cv2.TrackerMOSSE_create(), frame, boxforcars[-1])

        else:
            status = "Tracking"
            newbox = None
            success, boxes = trackers.update(frame)
            print(type(trackers))
            print(trackers)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



        frame = crossinglines(frame)

        cv2.imshow( 'Video', frame)
        if cv2.waitKey(1)&0xff==ord('q'):
            break
        
"""                
#################################   LICENSE PLATE DETECTION   #################################

                #Time 2
                start_time = time.time()

                input_dir  = 'tmp/output'
                output_dir = input_dir

                imgs_paths = glob('%s/*car.png' % input_dir)

                print('Searching for license plates using WPOD-NET')

                for i,img_path in enumerate(imgs_paths):

                    print('\t Processing %s' % img_path)

                    bname = splitext(basename(img_path))[0]
                    Ivehicle = cv2.imread(img_path)

                    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
                    side  = int(ratio*288.)
                    bound_dim = min(side + (side%(2**4)),608)
                    print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

                    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

                    if len(LlpImgs):
                        Ilp = LlpImgs[0]
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                        s = Shape(Llp[0].pts)

                        cv2.imwrite('%s/%s_lp.png' % (output_dir,n),Ilp*255.)
                        writeShapes('%s/%s_lp.txt' % (output_dir,n),[s])

                end_time = time.time()
                print("License plate detection time: " + str(end_time - start_time))

#################################   LICENSE PLATE OCR   #################################

                #Time 3
                start_time = time.time()

                input_dir  = b'tmp/output'
                output_dir = input_dir

                ocr_threshold = .4

                ocr_weights = b'data/ocr/ocr-net.weights'
                ocr_netcfg  = b'data/ocr/ocr-net.cfg'
                ocr_dataset = b'data/ocr/ocr-net.data'

                ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
                ocr_meta = dn.load_meta(ocr_dataset)

                imgs_paths = sorted(glob(b'%s/*lp.png' % output_dir))

                print('Performing OCR...')

                for i,img_path in enumerate(imgs_paths):

                    print('\tScanning %s' % img_path)

                    bname = basename(splitext(img_path)[0])

                    R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)


                    if len(R):

                        L = dknet_label_conversion(R,width,height)
                        L = nms(L,.45)

                        L.sort(key=lambda x: x.tl()[0])
                        lp_str = ''.join([chr(l.cl()) for l in L])
                        output_dir = 'tmp/output'
                        with open('%s/%s_str.txt' % (output_dir,n),'w') as f:
                           f.write(lp_str + '\n')

                        result = kazakhstan(lp_str)

                        if (result != ""):
                            print('-----------------------------------------------------------------')
                            print('\t\tLP: %s' % result)
                            print('-----------------------------------------------------------------')
                        
                        if len(lp_str) > 5: #Parameter to tune

                            print('-----------------------------------------------------------------')
                            print('\t\tLP: %s' % lp_str)
                            print('-----------------------------------------------------------------')
                            cv2.putText(frame, "Number: " + lp_str, (x1-5, y1-5),
                            	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    else:

                        print('No characters found')	

                end_time = time.time()
                print("License plate OCR time: " + str(end_time - start_time))

#################################   GENERATE OUTPUTS   #################################

                #Time 4
                start_time = time.time()

                YELLOW = (  0,255,255)
                RED    = (  0,  0,255)

                input_dir = 'frame.jpg'
                output_dir = 'tmp/output'

                I = cv2.imread(input_dir)

                detected_cars_labels = '%s/%s_cars.txt' % (output_dir, n)

                Lcar = lread(detected_cars_labels)

                sys.stdout.write('%s' % n)
                j = 0
                if Lcar:

                    for i,lcar in enumerate(Lcar):

                        draw_label(I,lcar,color=YELLOW,thickness=3)

                        lp_label 		= '%s/%s_%dcar_lp.txt'		% (output_dir,n,i)
                        lp_label_str 	= '%s/%s_%dcar_lp_str.txt'	% (output_dir,n,i)

                        if isfile(lp_label):

                            Llp_shapes = readShapes(lp_label)
                            pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                            ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                            draw_losangle(I,ptspx,RED,3)

                            if isfile(lp_label_str):
                                with open(lp_label_str,'r') as f:
                                    lp_str = f.read().strip()
                                llp = Label(0,tl=pts.min(1),br=pts.max(1))
                                write2img(I,llp,lp_str)

                                sys.stdout.write(',%s' % lp_str)

                cv2.imwrite('%s/%s_output.png' % (output_dir,n),I)
                sys.stdout.write('\n')

                end_time = time.time()
                print("Generate outputs time: " + str(end_time - start_time))

                shutil.rmtree('tmp/output/')

                cv2.imshow('Video', frame)

                if cv2.waitKey(1)&0xff==ord('q'):
                    break

                n += 1

                if writer is None:
                	#initialize our video write
                	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                	writer = cv2.VideoWriter("output_video.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

                writer.write(frame)"""
#writer.release()
print("Finished.")
cap.release()
