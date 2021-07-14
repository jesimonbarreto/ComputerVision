import numpy as np
import cv2
from matplotlib import pyplot as plt


def my_harris_implementation(image_gray, sobel_s = 3, k = 0.04):
    thres_r = 0.1
    w = 3
    size_i_x, size_i_y = image_gray.shape
    #gaussian filter
    image_gray = cv2.GaussianBlur(image_gray,(11, 11),3)
    #calculando derivada utilizando sobel
    I_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=sobel_s)
    I_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=sobel_s)
    #sum_w(I_x_2)
    
    I_x_2 = I_x * I_x
    I_y_2 = I_y * I_y
    I_x_y = I_x * I_y
    
    
    
    result = []
    
    for i in range(size_i_y):
        res = []
        for j in range(size_i_x):
            w_x = I_x_2[j:j+w,i:i+w]
            m_x2 = np.sum(w_x)
            w_y = I_x_2[j:j+w,i:i+w]
            m_y2 = np.sum(w_y)
            w_xy = I_x_y[j:j+w,i:i+w]
            m_xy = np.sum(w_xy)
            
            #m = [[m_x2,m_xy],[m_xy,m_y2]]
            
            r_w = m_x2*m_y2 - k * ((m_x2+m_y2) ** 2)
            res.append(r_w)
        result.append(res)
    
    my_harris = np.array(result)

    def my_non_max_supr(harris, w = 5):

        size_x, size_y = harris.shape

        final = np.zeros((size_x, size_y))

        for i in range(size_y-w):
            for j in range(size_x-w):
                win = harris[j:j+w,i:i+w]
                v = np.unravel_index(win.argmax(), win.shape)
                for sy in range(i, i+w):
                    for sx in range(j, j+w):
                        if (v[0] == sy - i and v[1] == sx - j and win[v[0],v[1]] > 375921500000 ):
                            final[sx,sy] = 1
                        else:
                            final[sx,sy] = 0
                        
                            
        return final
    
    final_result = my_non_max_supr(my_harris, w = 5)       
        
    def search_points(final_res):
        x,y = final_res.shape
        corners = []
        for i in range(x):
            for j in range(y):
                if final_res[i,j] == 1:
                    corners.append([i,j])
        return corners
    
    corners = search_points(final_result)
    
    return corners


filename = 'x_t.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
#dst = cv.cornerHarris(gray,2,3,0.04) 
corners = my_harris_implementation(gray, sobel_s = 3, k = 0.04)        


print(len(corners))