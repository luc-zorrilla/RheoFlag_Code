from unicodedata import is_normalized

from numpy import imag
from image_analysis_functions import *

import plotly.express as px
import plotly.graph_objects as go

###################
##### Imports #####

### Import video

chlamy_video = Video(filename='C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Image_Analysis\\Input\\chlamy_video_kirsty.avi')

### Import ROI

ROI_matrix = ImportROI('full_ROI.png').astype(int)

##### Imports #####
###################



#####################
##### Filtering #####

w_ma=151
chlamy_video = chlamy_video.deMovingAverage(w_ma)

chlamy_video.Threshold()

morph_ope_list=[]
k_erosion=np.ones((2,2),np.uint8)
morph_ope_list.append(['erosion', k_erosion]) 

chlamy_video.MorphOps(morph_ope_list)

chlamy_video.Keep_ROI(ROI_matrix)

chlamy_video.ClusterThreshold(15)

chlamy_video.Skeletonize("skeleton") # Normal skeletonization

##### Filtering #####
#####################


############################################
##### Axonemes extraction and ordering #####

axoneme_left_video, axoneme_right_video = chlamy_video.ExtractAndOrder()

##### Axonemes extraction and ordering #####
############################################


#########################
##### Visualization #####

img_list = []

# plt.imshow(chlamy_video.data[10], interpolation="nearest")
# plt.plot(axoneme_left_video[10][:,0], axoneme_left_video[10][:,1], 'bx')
# plt.plot(axoneme_right_video[10][:,0], axoneme_right_video[10][:,1], 'gx')

# x1 = axoneme_left_video[10][:,0]
# y1 = axoneme_left_video[10][:,1]
# x2 = axoneme_right_video[10][:,0]
# y2 = axoneme_right_video[10][:,1]

# image = cv2.cvtColor(chlamy_video.data[10],cv2.COLOR_GRAY2BGR)
# # curve1 = np.column_stack((x1.astype(np.int32), y1.astype(np.int32)))
# # curve2 = np.column_stack((x2.astype(np.int32), y2.astype(np.int32)))
# # image = cv2.polylines(image, [curve1, curve2], False, (0,255,255))
# plt.imshow(image)
# plt.plot(axoneme_left_video[10][:,0], axoneme_left_video[10][:,1], color='blue' , marker= 'o', linestyle='dashed', linewidth=1, markersize=0.5)
# plt.plot(axoneme_right_video[10][:,0], axoneme_right_video[10][:,1], color='green' , marker= 'o', linestyle='dashed', linewidth=1, markersize=0.5)

# plt.show()

z = cv2.cvtColor(chlamy_video.data[10],cv2.COLOR_GRAY2BGR)
x = axoneme_left_video[10]
y = axoneme_right_video[10]
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on
fig.show() # Initially shows the figure


viewer.clear() # Clears the previous image
viewer.imshow(z) # Loads the new image
viewer.plot(x[:,0], x[:,1], color='blue' , marker= 'o', linestyle='dashed', linewidth=1, markersize=1)
viewer.plot(y[:,0], y[:,1], color='green' , marker= 'o', linestyle='dashed', linewidth=1, markersize=0.5)

fig.canvas.draw() # Draws the image to the screen
img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
        sep='')
img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# img is rgb, convert to opencv's default bgr
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print("image!")
plt.figure()
plt.imshow(img)
plt.show()

exit()
z = chlamy_video.data
x = axoneme_left_video
y = axoneme_right_video
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on
fig.show() # Initially shows the figure


# for i in range(len(x)):

#     viewer.clear() # Clears the previous image
#     viewer.imshow(z[i]) # Loads the new image
#     viewer.plot(x[i][:,0], x[i][:,1], color='blue' , marker= 'o', linestyle='dashed', linewidth=1, markersize=1)
#     viewer.plot(y[i][:,0], y[i][:,1], color='green' , marker= 'o', linestyle='dashed', linewidth=1, markersize=0.5)

#     # plt.savefig('image_' + str(i) + '_.png')
    
#     fig.canvas.draw() # Draws the image to the screen
#     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
#             sep='')
#     img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     # img is rgb, convert to opencv's default bgr
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


##### Visualization #####
#########################

###################
##### Testing #####

##### Testing #####
###################

#########################
####### Animation #######

# a = chlamy_video.data
# b = axoneme_left_video
# c = axoneme_right_video

# # fig = plt.figure()
# # ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# # line, = ax.plot([], [], lw=2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# #line, = ax.plot([], [], lw=2)
# line, = lines.Line2D(b[0][:,0], b[0][:,1])

# def init():
#     line.set_data([],[])
#     return line,

# def animate(i):
#     x = b[i][:,0]
#     y = b[i][:,1]
#     line.set_data(x, y)
#     return line,

# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=10, blit=True)

# # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()

####### Animation #######
#########################

##########################
##### Create a video #####

##### Create a video #####
##########################