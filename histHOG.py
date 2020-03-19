import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

img_original = cv2.cvtColor(cv2.imread("0060.jpg"),
                   cv2.COLOR_BGR2GRAY)

img_resized = cv2.resize(img_original, (440,280)) 

#img = cv2.imread('0002.jpg')

image = img_resized

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, feature_vector=True)

#print(fd)

#with open('Fitur\mobil30.txt', 'wb') as f:
#    np.savetxt(f, np.column_stack(fd), fmt='%1.10f')

#plt.plot(fd)

start_row1, start_col1 = 135, 34
end_row1, end_col1 = 194, 103
crop_img1 = img_resized[start_row1:end_row1, start_col1:end_col1]
fd1, hog_image = hog(crop_img1, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, feature_vector=True)
#print(fd1)
#plt.plot(fd)
#plt.show()
with open('Fitur\dan1.txt', 'wb') as f:
    np.savetxt(f, np.column_stack(fd1), fmt='%1.10f')
#cv2.imshow("cropped", crop_img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

start_row2, start_col2 = 158, 193
end_row2, end_col2 = 220, 257
crop_img2 = img_resized[start_row2:end_row2, start_col2:end_col2]
fd2, hog_image = hog(crop_img2, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, feature_vector=True)
with open('Fitur\dan2.txt', 'wb') as f:
    np.savetxt(f, np.column_stack(fd2), fmt='%1.10f')
#print(fd2)
#plt.plot(fd2)
#plt.show()
#cv2.imshow("cropped", crop_img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

start_row3, start_col3 = 164, 340
end_row3, end_col3 = 226, 404
crop_img3 = img_resized[start_row3:end_row3, start_col3:end_col3]
fd3, hog_image = hog(crop_img3, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, feature_vector=True)
with open('Fitur\dan3.txt', 'wb') as f:
    np.savetxt(f, np.column_stack(fd3), fmt='%1.10f')
#cv2.imshow("cropped", crop_img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

