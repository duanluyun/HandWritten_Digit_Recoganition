
import tensorflow as tf
from PIL import Image
import os


import LeNet_inference
import LeNet_Train
import Segment
import Clear

def imageprepare(path):
    res=[]
    file_list=os.listdir(path)
    file_list.sort()
    file_name=[path+"/"+i for i in file_list]
    for n in file_name:
        im = Image.open(n).convert('L')
        tv = list(im.getdata())
        tva = [(255-x)*1.0/255.0 for x in tv]
        res.append(tva)
    return res

def connect(p):
    n=len(p)
    i=-1
    wq=1
    sum=0
    while i>=-n:
        tmp=wq*p[i]
        sum+=tmp
        i-=1
        wq=wq*10
    print("recognize result:")
    print(sum)
    return sum


def imagecheck(path):
    Segment.segment(path)
    result=imageprepare("/home/sam/Downloads/imageprepare/tmp")
    x = tf.placeholder(tf.float32, [None, 784])
    xs = tf.reshape(x, [len(result), 28, 28, 1])
    y = LeNet_inference.inference(xs, False, None)
    variable_averages = tf.train.ExponentialMovingAverage(LeNet_Train.MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, '/home/sam/Documents/Python_work/LearningPython/tensorflow/MODEL_SAVE4/model.ckpt-49001')
        prediction = tf.argmax(y, 1)
        predint = prediction.eval(feed_dict={x: result}, session=sess)
    result=connect(predint)
    Clear.clear("/home/sam/Downloads/imageprepare/tmp")
    return result




