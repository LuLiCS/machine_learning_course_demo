import numpy as np
import struct


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


def distance(vec1,vec2):
    return np.linalg.norm(vec1 - vec2)


if __name__ == "__main__":
    train_img = loadImageSet("E:/python/mnist/train-images.idx3-ubyte")
    train_labels = loadLabelSet("E:/python/mnist/train-labels.idx1-ubyte")
    test_img = loadImageSet("E:/python/mnist/t10k-images.idx3-ubyte")
    test_labels = loadLabelSet("E:/python/mnist/t10k-labels.idx1-ubyte")
    matrix = [[0] * 1000 for i1 in range(1000)]
    predict = [0] * 1000
    true = 0
    false = 0
    for i in range(0, 1000):
        for j in range(0, 1000):
            matrix[i][j] = distance(train_img[i],train_img[j])
        array1 = np.argsort(matrix[i])

        label1 = train_labels[array1[1]]
        label2 = train_labels[array1[2]]
        label3 = train_labels[array1[3]]
        #print(train_labels[i])
        #print "1", label1
        #print "2", label2
        #print "3", label3
        if label1 != label2 and label2 != label3 and label1 != label3:
            predict[i] = label1
        elif label1 == label2:
            predict[i] = label1
        elif label3 == label2:
            predict[i] = label2
        elif label1 == label3:
            predict[i] = label1
        if predict[i] == train_labels[i]:
            true += 1
        else:
            false += 1
    rate = (float)(true)/(true + false)
    print "true:", true
    print "false:",false
    print "rate:", rate