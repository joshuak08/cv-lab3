from collections import defaultdict
import math
import numpy as np
import cv2

input_image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)

padded = np.pad(input_image, 1, mode='constant', constant_values=0)

def threshold(image, threshold):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] >= threshold:
                image[y][x] = 255
            else:
                image[y][x] = 0
    return image

def scaled(image):
    maximum = image.max()
    minimum = image.min()
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            output[y][x] = ((pixel-minimum)/(maximum-minimum))*255
    return output

def sobel(matrix, kernel):
    dx = np.zeros(shape=(input_image.shape[0], input_image.shape[1]))
    dy = np.zeros(shape=(input_image.shape[0], input_image.shape[1]))
    mag = np.zeros(shape=(input_image.shape[0], input_image.shape[1]))
    gradient = np.zeros(shape=(input_image.shape[0], input_image.shape[1]))
    offset = kernel.shape[0]//2
    for i in range(offset, matrix.shape[0]-offset):
        for j in range(offset, matrix.shape[1]-offset):
            sectioned = padded[i-offset:i+offset+1, j-offset:j+offset+1]
            dx[i-offset][j-offset] = convolution(sectioned, kernel)
            dy[i-offset][j-offset] = convolution(sectioned, kernel.T)
            # dx[i-1][j-1], dy[i-1][j-1], mag[i-1][j-1], gradient[i-1][j-1] = 
    mag = np.sqrt(dx**2 + dy**2)
    gradient = np.arctan(dy/(dx+np.exp(-10)))
    return dx, dy, mag, gradient

def convolution(matrix, kernel):
    output = 0
    kernel_y = kernel.shape[0]
    kernel_x = kernel.shape[1]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output += (matrix[i][j] * kernel[kernel_y-1-i][kernel_x-1-j])
    return output

def hough(magnitude, gradient, r_min, r_max):
    height, width = magnitude.shape[0], magnitude.shape[1]
    hough_space = np.zeros((height, width, r_max+1-r_min))

    for r in range(r_min, r_max+1):
        for y in range(height):
            for x in range(width):
                if magnitude[y][x] != 0:
                    x_minus = round(x - r*np.cos(gradient[y][x]))
                    y_minus = round(y - r*np.sin(gradient[y][x]))
                    x_add = round(x + r*np.cos(gradient[y][x]))
                    y_add = round(y + r*np.sin(gradient[y][x]))

                    if x_minus >= 0 and x_minus < width:
                        if y_minus >= 0 and y_minus < height:
                            hough_space[y_minus][x_minus][r-r_min] += 1
                        
                    if x_add >= 0 and x_add < width:
                        if y_add >= 0 and y_add < height:
                            hough_space[y_add][x_add][r-r_min] += 1
    return hough_space

def draw_circle(image, threshold_hough_space, hough_space, r_min, k=10):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
    white_pixels = []
    for y in range(threshold_hough_space.shape[0]):
        for x in range(threshold_hough_space.shape[1]):
            if threshold_hough_space[y,x] == 255:
                white_pixels.append([x, y])

    ret, label, center=cv2.kmeans(np.array(white_pixels, dtype=np.float32), 10, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    output = image.copy()
    for c in center:
        x, y = round(c[0]), round(c[1])
        r = hough_space[y,x].argmax(axis=0) + r_min
        output = cv2.circle(output, (x,y), 40, (255,0,0), 2)
    
    return output

weights = np.array([[1],[2],[1]]) # shape (3,1)
d = np.array([[-1,0,1]]) # shape (1,3)
dx_kernel = np.dot(weights,d)

dx, dy, magnitude, gradient = sobel(padded, dx_kernel)

r_min = 36
r_max = 44

hough_space = hough(threshold(magnitude, 50), gradient, r_min, r_max)

for i in range(r_min, r_max+1):
    cv2.imwrite(f'r_hough_space_{i}.png', scaled(hough_space[:,:,i-r_min]))

allSum = np.sum(hough_space, axis=2)
cv2.imwrite('r_hough_space.png', scaled(allSum))

threshold_hough_space = threshold(scaled(allSum), 46)

output = draw_circle(input_image, threshold_hough_space, hough_space, r_min, 10)

cv2.imwrite('circle detection coins1.png', output)
cv2.imshow('test', output)
cv2.waitKey()
cv2.destroyAllWindows()
