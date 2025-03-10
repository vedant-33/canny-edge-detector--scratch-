import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

def gaussian_blur(image, kernel_size=5, sigma=1):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss) #2D gaussian
    kernel = kernel / np.sum(kernel) # normalise

    return convolve2d(image, kernel,mode='same', boundary='symm')


def non_max_suppression(magnitude, direction):
    rows, cols= magnitude.shape
    suppressed_image= np.zeros((rows,cols), dtype= np.float32)

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            angle= direction[i,j]
            nbr1, nbr2= 255,255

            # sort of descretised the gradient angle
            if(0<= angle<22.5 or 157.5 <=angle<180):
                nbr1= magnitude[i,j+1]
                nbr2= magnitude[i,j-1]
            elif(22.5 <=angle<67.5):
                nbr1= magnitude[i+1,j+1]
                nbr2= magnitude[i-1,j-1]
            elif(67.5<= angle< 112.5):
                nbr1= magnitude[i-1,j]
                nbr2= magnitude[i+1,j]
            elif(112.5 <=angle<157.5):
                nbr1= magnitude[i-1,j+1]
                nbr2= magnitude[i+1,j-1]
            
            # is i,j the strongest pixel
            if magnitude[i,j] >= nbr1 and magnitude[i,j]>= nbr2:
                suppressed_image[i,j] =magnitude[i,j]
            else:
                suppressed_image[i,j]=0

    return suppressed_image
    

def hysteresis(strong_edges, weak_edges):
    rows,cols= strong_edges.shape
    final_edges = strong_edges.copy()

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if weak_edges[i,j] == 50:
                # check nbrs of i,j
                if(strong_edges[i-1,j]==255 or strong_edges[i-1,j-1]==255 or strong_edges[i-1,j+1]==255 or strong_edges[i+1,j]==255 or strong_edges[i+1,j-1]==255 or strong_edges[i+1,j+1]==255 or strong_edges[i,j-1]==255 or strong_edges[i,j+1]==255):
                    final_edges[i,j] =255
                else:
                    final_edges[i,j]=0
    
    return final_edges
    

def canny_edge_scratch(img_path):
    image= Image.open(img_path).convert('L') #grayscale
    image= np.array(image,dtype= np.float32)

    smoothed_image= gaussian_blur(image) # gaussian smoothing

    sobel_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_Y= np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    grad_X= convolve2d(smoothed_image, sobel_X, mode='same', boundary='symm') # array of all pixel values
    grad_Y = convolve2d(smoothed_image, sobel_Y,mode='same', boundary='symm')

    magnitude = np.hypot(grad_X, grad_Y) # gradient
    direction = np.arctan2(grad_Y, grad_X) * (180/np.pi) # gradient orientation
    direction[direction < 0] += 180 # normalise all angles

    suppressed_image= non_max_suppression(magnitude, direction)
    
    #thresholding
    higher_threshold= np.max(suppressed_image) * 0.2
    lower_threshold= higher_threshold * 0.6
    strong, weak= np.zeros_like(suppressed_image), np.zeros_like(suppressed_image)

    strong[suppressed_image >= higher_threshold] =255
    weak[(suppressed_image<higher_threshold) & (suppressed_image>= lower_threshold)] = 50

    edges= hysteresis(strong, weak)

    return edges

edges= canny_edge_scratch('C:/Users/dhruv/OneDrive/Desktop/Vedant - personal/Documents/vedant (2).jpeg')

plt.imshow(edges, cmap= 'gray')
plt.show()