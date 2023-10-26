import numpy as np
import cv2

image = cv2.imread('coins1.png')
padded = np.pad(image, 1, mode='constant', constant_values=0)

def convolve(img, kern):
    height = img.shape[0]
    width = img.shape[1]
    bottom = 10000000 #range
    top = -10000000 #range
    img_float = np.zeros(img.shape)
    print( height - (kern.shape[0] // 2))
    print( width - (kern.shape[1] // 2))
    for i in range(kern.shape[0] // 2, height - (kern.shape[0] // 2)):
        for j in range(kern.shape[1] // 2, width - (kern.shape[1] // 2)):
            values = getKernelPixelValues(img, kern, j, i)
            # print(values)
            px = [conv(values[:,:,0], kern), 
                        conv(values[:,:,1], kern), 
                        conv(values[:,:,2], kern)]

            largeval = max(px[0], px[1], px[2])
            smallval = min(px[0], px[1], px[2])
            #print(largeval)
            bottom = min(smallval, bottom)
            top = max(largeval, top)
            img_float[i][j] = px

    return rerange(img_float, bottom, top)

def getKernelPixelValues(image, kernel, j, i):
    return image[i-1:i+2, j-1:j+2,:]

def conv(matrix, kernel):
    output = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output += matrix[i][j] * kernel[2-i][2-j]
    return output*1/8

def rerange(img, bottom, top):
    # print(bottom, top)
    range_number = top - bottom
    # print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            img[i][j] = (img[i][j] * (255 / range_number)) - bottom
    return img

dx_kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
output = convolve(image, dx_kernel)

cv2.imshow('test', output)

cv2.waitKey()
cv2.destroyAllWindows()