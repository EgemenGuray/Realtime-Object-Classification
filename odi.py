import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import scipy.misc
from numpy import asarray
import torch
from torch.autograd import Variable
from PIL import Image
import numpy


def detect(frame, net, transform):
    height = frame.shape[0]
    width = frame.shape[1]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of calsses, number of occurence, (score, x0, y0, x1, y1)]
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] > 0.7:
            pt = (detections[0,i,j,1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0, 0), 2)
            cv2.putText(frame, labelmap[i-1] +" " +  str(round(detections[0,i,j,0],4)), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),1, cv2.LINE_AA)
            j += 1
            
    return frame


        





def retImage(path= 'image.jpeg'):
    net = build_ssd('test')
    net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc : storage))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    
    #frame = cv2.imread(path)
    img = Image.open(path)
    frame = numpy.array(img)
    retval = detect(frame, net, transform)
    new_im = Image.fromarray(retval)
    new_im.save("numpy_altered_sample2.png")
    print('Done')
      



retImage('test3.jpg')
    