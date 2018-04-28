# HandWritten_Digit_Recoganition
In this work we focused on two classifiers.Convolutional Neural Network(CNN) and Support Vector Machine(SVM) for offline handwriting digits recoganition.A convolutional network is beneficial for extracting features information. Currently,support vector machines(SVMs) are extensively used for recognizing handwritten digits because of their high performance in pattern recognition. SVMs are binary classfiers based on the underlying structural risk minimization principle. They procced by mapping data into a high dimensional dot product space via a kernel function.In this space an optimal hyperplane,that maximizes the margin of separation between the two classes,is calculated.

## Requirements:
1.Numpy==1.12.1

2.tensorflow==1.4.0

3.Opencv

4.Python==3.5

5.Pillow==5.0.0

6.Sklearn

7.PyMySQL==0.7.10

8.MySQl

9.opencv_python=3.3.0

10.pandas

11.matplotlib


## HandWritten Multi-digit String Segmentation
Usually, the recognition of the segmented digits is an easier task compared to segmentation and recognition of a multi-digit string.  In this work, we will use opencv  segment a handwritten multi-digit string image and recognize the segmented digits.


## Steps:
1.load image
``img=cv.read(path)``

![Source Image](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/1.png)

2.preprocessint the image

(1)Convert image to grayscale :
Assigning each pixel to a value of the range of monochromatic shades from black to white to represent its amount of light.

(2)Binarise image:
Assigning each pixel to only two possible colors typically black and white.

![gray Image](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/2.png)

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

![Annotated the contour](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/3.png)


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

6.Add black border to each digit, this increases the accuracy of classification
```python
h, w = im.size
newH = w // 2 - h // 2
imgEmpty = Image.new('L', (w, w), 0)
imgEmpty.paste(im, (newH, 0))
imageResize = imgEmpty.resize((28, 28), Image.ANTIALIAS)
```
---
```python
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

```

![gray Image](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/7.png)

## HandWritten Digit Recoganition(CNN)

## Steps:

### (1)Sort images in the directory
### (2)get data from each image
### (3)reshape it to 1*784
```python
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
```

### (4)Recoganize the digit with CNN(based on the mnist dataset)

You can see the details in LeNet_inference.py and LeNet_train.py

### (5) get the original digit
```python
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
```


## HandWritten Digit Recoganition(SVM)

## Steps:

### 1.load the Original dataset
```python
if __name__=='__main__':
    pddata_train=pd.read_csv('.....',header=None)
    pddata_test=pd.read_csv('......',header=None)
    x_train=pddata_train.loc[:,pddata_train.columns!=64]
    y_train=pddata_train.loc[:,pddata_train.columns==64]

    x_train=np.array(x_train)
    images_train=x_train.reshape([-1,8,8])

    x_test= pddata_test.loc[:, pddata_train.columns != 64]
    y_test = pddata_test.loc[:, pddata_train.columns == 64]

    x_test = np.array(x_test)
    images_test = x_test.reshape([-1, 8, 8])
    plt.figure(figsize=(100,50))
    for index,image in enumerate(images_train[:16]):
        plt.subplot(4,8,index+1)
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title('the ture number is %d'%(y_train.loc[index]))
    for index,image in enumerate(images_test[:16]):
        plt.subplot(4,8,index+17)
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title('the true number is %d'%(y_test.loc[index]))
    plt.show()

```

![gray Image](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/5.png)

### 2.Uses classical SVM with RBF kernel. The drawback of this solution is rather long training on big datasets, although the accuracy with good parameters is high.The accuracy is 98.27%

```python
 clf=SVC(C=1,kernel='rbf',gamma=0.001)
    clf.fit(x_train,y_train.values.ravel())
    y_hat=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_hat)
    print('the accuracy is %f'%(accuracy) )
    plt.figure(figsize=(100,50))
    y_hat_err=y_hat[y_hat!=y_test.values.ravel()]
    y_test_err=y_test[y_hat!=y_test.values.ravel()].values.ravel()
    err_images=images_test[y_hat!=y_test.values.ravel()]

    for index,err_image in enumerate(err_images):

        plt.subplot(4,8,index+1)
        plt.imshow(err_image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title('y_hat=%d, y=%d'%(y_hat_err[index],y_test_err[index]))


```
### The error classification from the test dataset

![gray Image](https://github.com/duanluyun/HandWritten_Digit_Recoganition/raw/master/Image/6.png)
