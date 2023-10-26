import numpy as np
import cv2

image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)

weights = np.array([[1],[2],[1]]) # shape (3,1)
d = np.array([[-1,0,1]]) # shape (1,3)
dx_kernel = np.dot(weights,d)
dy_kernel = np.dot(d.T, weights.T)


def convolution(matrix, kernel):
    output = 0
    kernel_y = kernel.shape[0]
    kernel_x = kernel.shape[1]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output += (matrix[i][j] * kernel[kernel_y-1-i][kernel_x-1-j])
    return output


def sobel(matrix, kernel):
    dx = np.zeros(shape=(image.shape[0], image.shape[1]))
    dy = np.zeros(shape=(image.shape[0], image.shape[1]))
    mag = np.zeros(shape=(image.shape[0], image.shape[1]))
    gradient = np.zeros(shape=(image.shape[0], image.shape[1]))
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

    
padded = np.pad(image, 1, mode='constant', constant_values=0)

# for i in range(1, padded.shape[0]-1):
#     for j in range(1, padded.shape[1]-1):
#         sectioned = padded[i-1:i+2, j-1:j+2]
#         dx[i-1][j-1], dy[i-1][j-1], mag[i-1][j-1], gradient[i-1][j-1] = sobel(sectioned)

def scaled(image):
    maximum = image.max()
    minimum = image.min()
    scale = maximum - minimum
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            output[y][x] = ((pixel-minimum)/(maximum-minimum))*255
    return output

fx, fy, magnitude, gradient = sobel(padded, dx_kernel)

print('angle', gradient.min(), gradient.max())


# cv2.imwrite("rewrite_dx.png", scaled(fx))
# cv2.imwrite("rewrite_dy.png", scaled(fy))
# cv2.imwrite("rewrite_mag.png", scaled(magnitude))
cv2.imwrite("nonscale_gradient.png", gradient)

cv2.imshow('gradient', gradient.astype('uint8'))
cv2.waitKey()
cv2.destroyAllWindows()
