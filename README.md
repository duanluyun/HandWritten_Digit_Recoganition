# HandWritten_Digit_Recoganition
In this work we focused on two classifiers.Convolutional Neural Network(CNN) and Support Vector Machine(SVM) for offline handwriting digits recoganition.A convolutional network is beneficial for extracting features information. Currently,support vector machines(SVMs) are extensively used for recognizing handwritten digits because of their high performance in pattern recognition. SVMs are binary classfiers based on the underlying structural risk minimization principle. They procced by mapping data into a high dimensional dot product space via a kernel function.In this space an optimal hyperplane,that maximizes the margin of separation between the two classes,is calculated.

## HandWritten Multi-digit String Segmentation
Usually, the recognition of the segmented digits is an easier task compared to segmentation and recognition of a multi-digit string.  In this work, we will use opencv  segment a handwritten multi-digit string image and recognize the segmented digits.
## Steps:
1.load image
``img=cv.read(path)``

![Alt text](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/1.png)

2.preprocessint the image

(1)Convert image to grayscale :
Assigning each pixel to a value of the range of monochromatic shades from black to white to represent its amount of light.

(2)Binarise image:
Assigning each pixel to only two possible colors typically black and white.

```python
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
```
3.Annotate the contour
```python
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

```
4.Segmet Multi-digit to individual digits and on the same time resize them
```python
def divide(img,thres):
    #------------------------divide digit----------------------------------------------------
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

```

```python
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

```

5.Sort the contour from left to right
```python
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

```
