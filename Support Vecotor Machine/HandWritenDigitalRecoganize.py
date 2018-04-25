import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

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

    plt.show()









