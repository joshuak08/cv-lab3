import numpy as np
import cv2
from scipy import signal

image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)

dx_kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
dy_kernel = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
test_kernel = np.arange(9).reshape((3,3))+1

def convolution(matrix, kernel):
    output = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output += matrix[i][j] * kernel[2-i][2-j]
    return output/8

def scaled(image):
    maximum = image.max()
    minimum = image.min()
    scale = maximum - minimum
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            output[y][x] = ((pixel-minimum)/(maximum-minimum))*255

    return output

# def correlation(matrix, kernel):
#     return np.sum(matrix*kernel, axis=(0,1))

def sobel(matrix):
    # print(matrix)
    outputX = 0
    outputY = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            outputX += (matrix[i][j]  * hx[2-i][2-j])
            outputY += (matrix[i][j]  * hy[2-i][2-j])
    mag = np.sqrt(outputX**2 + outputY**2)
    gradient = np.arctan(outputY/(outputX+0.000000001))
    # print(gradient)
    return outputX, outputY, mag, gradient
test_output = np.zeros(shape=(image.shape[0], image.shape[1]))
corr = np.zeros(shape=(image.shape[0], image.shape[1]))
padded = np.pad(image, 1, mode='constant', constant_values=0)
for i in range(padded.shape[0]-2):
    for j in range(padded.shape[1]-2):
        # if image[i:i+3, j:j+3].shape == (3,2):
        #     print(i, j)
        test_output[i][j] = convolution(image[i:i+3, j:j+3], dx_kernel)
        # corr[i][j] = correlation(image[i:i+3, j:j+3], dx_kernel)
scale = scaled(test_output)
cv2.imshow('test', scaled(test_output).astype('uint8'))
cv2.imwrite('test.jpg', test_output)
cv2.imwrite('scaled.jpg', scale)
cv2.imshow('scaled', scale)
# cv2.imshow('correlation', corr)

cv2.waitKey()
cv2.destroyAllWindows()

# print(signal.fftconvolve(test, dx_kernel, mode='valid'))
# dx = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int8)
# dy = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int8)
# mag = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int8)
# gradient = np.zeros(shape=(image.shape[0], image.shape[1]))

# for i in range(1, padded.shape[0]-1):
#     for j in range(1, padded.shape[1]-1):
#         sectioned = padded[i-1:i+2, j-1:j+2]
#         dx[i-1][j-1], dy[i-1][j-1], mag[i-1][j-1], gradient[i-1][j-1] = sobel(sectioned)

# cv2.imshow("dx", dx)
# cv2.imshow("dy", dy)
# cv2.imshow("mag", mag)
# cv2.imshow("gradient", gradient)

# cv2.waitKey()
# cv2.destroyAllWindows()
