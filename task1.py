import numpy as np
import cv2

image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)

weights = np.array([[1],[2],[1]]) # shape (3,1)
d = np.array([[-1,0,1]]) # shape (1,3)
hx = np.dot(weights,d)
hy = np.dot(d.T, weights.T)

def sobel(matrix):
    outputX = 0
    outputY = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            outputX += (matrix[i][j]  * hx[2-i][2-j])
            outputY += (matrix[i][j]  * hy[2-i][2-j])
    mag = np.sqrt(outputX**2 + outputY**2)
    gradient = np.arctan(outputY/(outputX+np.exp(-10)))
    return outputX, outputY, mag, gradient

    
padded = np.pad(image, 1, mode='constant', constant_values=0)
dx = np.zeros(shape=(image.shape[0], image.shape[1]))
dy = np.zeros(shape=(image.shape[0], image.shape[1]))
mag = np.zeros(shape=(image.shape[0], image.shape[1]))
gradient = np.zeros(shape=(image.shape[0], image.shape[1]))

for i in range(1, padded.shape[0]-1):
    for j in range(1, padded.shape[1]-1):
        sectioned = padded[i-1:i+2, j-1:j+2]
        dx[i-1][j-1], dy[i-1][j-1], mag[i-1][j-1], gradient[i-1][j-1] = sobel(sectioned)

def scaled(image):
    maximum = image.max()
    minimum = image.min()
    scale = maximum - minimum
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            output[y][x] = ((pixel-minimum)/(maximum-minimum))*255

    return output

cv2.imwrite("ori_dx.png", scaled(dx))
cv2.imwrite("ori_dy.png", scaled(dy))
cv2.imwrite("ori_mag.png", scaled(mag))
cv2.imwrite("ori_gradient.jpg", scaled(gradient))

# cv2.waitKey()
# cv2.destroyAllWindows()
