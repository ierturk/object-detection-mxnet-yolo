import mxnet as mx

from  data.cv2Iterator import CameraIterator
import logging
import cv2
import sys
import numpy as np
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv import model_zoo, data, utils

def load_img(frame, short=160, max_size=1024, stride=32, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mx.nd.array(frame)
    img = timage.resize_short_within(img, short, max_size, mult_base=stride)
    # orig_img = img.asnumpy().astype('uint8')
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    return img.expand_dims(0)

def draw_detection(frame, box, score, classID, class_names):

    (x0, y0, x1, y1) = box
    klass_name = class_names[int(classID)]
    h = frame.shape[0]
    w = frame.shape[1]
    p0 = tuple(map(int, (x0/160*h, y0/224*w)))
    p1 = tuple(map(int, (x1/160*h, y1/224*w)))
    # logging.info("detection: %s %s", klass_name, score)
    cv2.rectangle(frame, p0, p1, (0,0,255), 1)
    tp0 = (p0[0], p0[1]-5)
    draw_text = "{} {}".format(klass_name, score)
    cv2.putText(frame, draw_text, tp0, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,0,255))


def run_camera():

    logging.info("Camera iterator test {}".format(0), 2)
    iter = CameraIterator(frame_resize= 1.0)

    net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=mx.cpu(0))

    for frame in iter:
        x = load_img(frame)

        class_IDs, scores, bounding_boxs = net(x)
        
        for i in range(0, 100):
            # logging.info("Class ID {}".format(scores[0, i, 0]))

            if scores[0,i,0].asscalar() > 0.4:
                draw_detection(frame, bounding_boxs[0, i, :].asnumpy(), scores[0, i, :].asscalar(),
                               class_IDs[0, i, :].asscalar(), net.classes)

        cv2.imshow('frame', frame)

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')

    run_camera()

    return 0

if __name__ == '__main__':
    sys.exit(main())

