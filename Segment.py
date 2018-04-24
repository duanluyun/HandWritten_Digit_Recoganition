import numpy as np
import cv2 as cv
from PIL import Image

"""当出现弹窗时，按ＥＳＣ程序继续执行不保存中间结果，按Ｓ键保存中间结果并继续执行"""

def preprocessing(img):
    # -------------------preprocessing---------------------------
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bulre = cv.GaussianBlur(gray, (5, 5), 0)
    thres = cv.adaptiveThreshold(bulre, 255, 1, 1, 11, 2)
    cv.namedWindow("thres")
    cv.imshow("thres", thres)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
    return thres


def annotate_contour(img,thres):
    # --------------------annotate contours------------------------
    image, contours, hierarchy = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_annotated = img
    for cont in contours:
        if cv.contourArea(cont) > 20 :
            x, y, w, h = cv.boundingRect(cont)
            if h > 5:
                contour_annotated = cv.rectangle(contour_annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.namedWindow("contour_annotated")
    cv.imshow("contour_annotated", contour_annotated)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()


def divide(img,thres):
    #------------------------dividee digit----------------------------------------------------
    image, contours, hierarchy = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    conts=[]

    for cont in contours:
        if cv.contourArea(cont) > 20:
            [x, y, w, h] = cv.boundingRect(cont)
            if h > 5:
                conts.append([x,y,w,h])
    sort(conts)
    j = 1
    for i in range(len(conts)):
        tmp=thres[conts[i][1]:conts[i][1]+conts[i][3],conts[i][0]:conts[i][0]+conts[i][2]]
        tmp=np.array(tmp)
        tmp=resize(tmp)
        tmp.save("/home/sam/Downloads/imageprepare/tmp/"+str(j)+".png")
        j+=1



def sort(conts):
    #--------------------sort conts-------------------------------------
    for i in range(len(conts)):
        min=conts[i]
        j=i+1
        for j in range(len(conts)):
            if conts[j][0]>conts[i][0]:
                tmp=conts[i]
                conts[i]=conts[j]
                conts[j]=tmp
    return conts



def resize(cj):
    #----------------------resize(28*28) and threshold(175,255)-----------------------------
    im = Image.fromarray(cj)
    h, w = im.size
    newH = w // 2 - h // 2
    imgEmpty = Image.new('L', (w, w), 0)
    imgEmpty.paste(im, (newH, 0))
    imageResize = imgEmpty.resize((28, 28), Image.ANTIALIAS)
    imageResize0255 = imageResize.point(lambda x: 0 if x > 175 else 255)
    return imageResize0255



def segment(path):
    img = cv.imread(path)
    cv.namedWindow("src")
    cv.imshow("src",img)
    k=cv.waitKey(0)
    if k==27:
        cv.destroyAllWindows()
    # -------------------preprocessing---------------------------
    thres=preprocessing(img)
    #--------------------annotate contours------------------------
    annotate_contour(img,thres)
    #--------------------divide,sort and resize-----------------------------
    divide(img,thres)





