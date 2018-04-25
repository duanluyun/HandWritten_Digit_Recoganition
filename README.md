# HandWritten_Digit_Recoganition
In this work we focused on two classifiers.Convolutional Neural Network(CNN) and Support Vector Machine(SVM) for offline handwriting digits recoganition.A convolutional network is beneficial for extracting features information. Currently,support vector machines(SVMs) are extensively used for recognizing handwritten digits because of their high performance in pattern recognition. SVMs are binary classfiers based on the underlying structural risk minimization principle. They procced by mapping data into a high dimensional dot product space via a kernel function.In this space an optimal hyperplane,that maximizes the margin of separation between the two classes,is calculated.

## HandWritten Multi-digit String Segmentation
Usually, the recognition of the segmented digits is an easier task compared to segmentation and recognition of a multi-digit string.  In this work, we will use opencv  segment a handwritten multi-digit string image and recognize the segmented digits.
## Steps:
1.load image
``img=cv.read(path)``


2.preprocessint the image

(1)Convert image to grayscale :
Assigning each pixel to a value of the range of monochromatic shades from black to white to represent its amount of light.

(2)Binarise image:
Assigning each pixel to only two possible colors typically black and white.

```
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
