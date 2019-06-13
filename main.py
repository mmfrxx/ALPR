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

def russia(result1):
    result1.replace(" ", "")
    output = result1
    valid = True
    j = 0
    
    if len(result1) > 9 or len(result1) < 8:
        
        return ""
        
    for i in result1:
        if j == 0 or j == 4 or j == 5:
            if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                return ""
        if j == 1 or j == 2 or j == 3 or j == 6 or j == 7 or j == 8:
            if not (ord(i) >= 48 and ord(i) <= 57):
                return ""
        if j == 8:
            if not ((ord(i) >= 48 and ord(i) <= 57) or (ord(i) <= 32 and ord(i) >= 0)):
                return ""
        j += 1

    return output

def kazakhstan(result1):

    output = result1
    valid = True
    j = 0

    if len(result1) != 7:

        if len(result1) != 8:
            
            return ""
            
        for i in output:
            if j == 0 or j == 1 or j == 2 or j == 6:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    return ""
            if j == 3 or j == 4 or j == 5:
                if not((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                    return "" 
            if j == 7:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    return ""
            j += 1

        return output

    else:
        
        for i in result1:
            if j == 0 or j == 4 or j == 5 or j == 6:
                if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                    return ""
            if j == 1 or j == 2 or j == 3:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    return ""
            j += 1

        return output

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

def detectCarRegion(I,label,bg=0.5):

    wh = np.array(I.shape[1::-1])

    ch = I.shape[2] if len(I.shape) == 3 else 1
    tl = np.floor(label.tl()*wh).astype(int)
    br = np.ceil (label.br()*wh).astype(int)
    outwh = br-tl

    if np.prod(outwh) == 0.:
        return None

    outsize = (outwh[1],outwh[0],ch) if ch > 1 else (outwh[1],outwh[0])
    if (np.array(outsize) < 0).any():
        pause()
    Iout  = np.zeros(outsize,dtype=I.dtype) + bg

    offset  = np.minimum(tl,0)*(-1)
    tl      = np.maximum(tl,0)
    br      = np.minimum(br,wh)
    wh      = br - tl

    Iout[offset[1]:(offset[1] + wh[1]),offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1],tl[0]:br[0]]

    return Iout

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("cam2.avi")
    n = -1
    previous = ""
    writer = None
    while True:

        n += 1

        ret, frame = cap.read()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[INFO] {} total frames in video".format(total))
        frame = frame[:] #Parameter to tune
        if(total-n<21):break
        if ret == False:
            break

        if(n==1000): break
        try:

            print("Frame number #" + str(n))
            if n % 20 == 0: #Parameter to tune
        
                cv2.imwrite('frame.jpg', frame)
#################################   VEHICLE DETECTION   #################################

                #Time 1
                start_time = time.time()

                input_dir = b'frame.jpg'
                output_dir = 'tmp/output'

                if not isdir(output_dir):
                    makedirs(output_dir)

                print('Searching for vehicles using YOLO...')

                print('Scanning frame #%d' % n)

                R,img_shape = detect(vehicle_net, vehicle_meta, input_dir, thresh=vehicle_threshold)

                R = [r for r in R if r[0] in [b'car',b'bus']]

                if len(R) == 0:
                    continue

                print('\t\t%d cars found' % len(R))

                if len(R):

                    Iorig = frame
                    WH = np.array(Iorig.shape[1::-1],dtype=float)
                    Lcars = []

                    for i,r in enumerate(R):

                        cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                        print("HERE")
                        print(type(cx))
                        print(cx,cy)
                        print(type(w))
                        print(w)

                        x1 = int((cx-w/2)*np.concatenate((WH,WH))[0])
                        x2 = int((cx+w/2)*np.concatenate((WH,WH))[0])
                        y1 = int((cy-h/2)*np.concatenate((WH,WH))[0])-200
                        y2 = int((cy)*np.concatenate((WH,WH))[0])-200
                        tl = np.array([cx - w/2., cy - h/2.])
                        br = np.array([cx + w/2., cy + h/2.])
                        label = Label(0,tl,br)
                        Icar = crop_region(Iorig,label)

                        Lcars.append(label)
                        print("HERE")
                        print(type(label))
                        print(label)
                        print(Lcars)
                        
                        cv2.imwrite('%s/%s_%dcar.png' % (output_dir,n,i),Icar)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0))
                    lwrite('%s/%s_cars.txt' % (output_dir,n),Lcars)

                end_time = time.time()
                print("Vehicle detection time: " + str(end_time - start_time))

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

                writer.write(frame)

                continue
            
            else:
                continue

        except:
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)
    writer.release()
    print("Finished.")
    cap.release()
