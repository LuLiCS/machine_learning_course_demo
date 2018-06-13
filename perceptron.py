import numpy as np
import struct
import matplotlib.pyplot as plt

def show(num,true,predict):
    filename = 'E:/python/mnist/t10k-images.idx3-ubyte'
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 784*num
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')

    im = np.array(im)
    im = im.reshape(28, 28)

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.title("true:"+str(true)+"predict:"+str(predict))
    plt.show()

def loadImageSet(filename):
    print "load image set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print "head,", head

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print "load imgs finished"
    return imgs


def loadLabelSet(filename):
    print "load label set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print "head,", head
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print 'load label finished'
    return labels


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=1):
        self.eta = eta
        self.n_iter = n_iter
        self.b = 0
        w1 = np.zeros(784)
        self.w = np.resize(w1,(1,784))
    def fit(self, X, y):
        if y == -1:
            if (np.dot(self.w,np.transpose(X))+self.b)>= 0:
                self.w += X*self.eta*y
                self.b += self.eta*y
        elif y == 1:
            if (np.dot(self.w,np.transpose(X))+self.b)<= 0:
                self.w += X*self.eta*y
                self.b += self.eta*y
        return self

    def predict(self, xi):
        if np.dot(self.w,np.transpose(xi))+self.b >= 0:
            return 1
        else:
            return -1


if __name__ == "__main__":
    train_img = loadImageSet("E:/python/mnist/train-images.idx3-ubyte")
    train_labels = loadLabelSet("E:/python/mnist/train-labels.idx1-ubyte")
    test_img = loadImageSet("E:/python/mnist/t10k-images.idx3-ubyte")
    test_labels = loadLabelSet("E:/python/mnist/t10k-labels.idx1-ubyte")
    ppn=Perceptron(eta=0.1,n_iter=1)
    checkrate = [0]*50
    show_x=np.linspace(1,50)
    for it in range(0,50):
        for j in range(0,60000):
            if train_labels[j] == 0:
                ppn.fit(X=train_img[j],y=-1)
            elif train_labels[j] == 7:
                ppn.fit(X=train_img[j], y=1)
        true = 0
        false = 0
        for i in range(0,10000):
            if test_labels[i] == 0 or test_labels[i] == 7:
                if test_labels[i] == 0 and ppn.predict(xi=test_img[i]) == -1:
                    true += 1
                if test_labels[i] == 7 and ppn.predict(xi=test_img[i]) == 1:
                    true += 1
                if test_labels[i] == 7 and ppn.predict(xi=test_img[i]) == -1:
                    false += 1
                    if it == 0:
                        show(i,7,0)
                if test_labels[i] == 0 and ppn.predict(xi=test_img[i]) == 1:
                    false += 1
                    if it == 0:
                        show(i,0,7)
        rate = (float)(true) / (true + false)
        checkrate[it] = rate;
        print "iteration:",it+1,"true:",true
        print "iteration:",it+1,"false:",false
        print "iteration:",it+1,"rate:",rate
    plt.plot(show_x, checkrate)
    plt.show()