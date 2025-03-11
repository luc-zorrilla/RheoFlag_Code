import numpy as np
import plotly.express as px
from skimage import io
from video_analysis_functions import *

### Convert a set of tiff files into a stack of tiff files

# import glob
# import tifffile

# with tifffile.TiffWriter('Input_Stack.tif') as stack:
#     for filename in glob.glob('Input/*.tiff'):
#         stack.save(
#             tifffile.imread(filename), 
#             # photometric='minisblack', 
#             contiguous=True
#         )

### Open a tif-stack file

original_data = io.imread('Input_Stack.tif')
print("data.shape", original_data.shape)

### Prepare the data

## Convert into grayscale
data = original_data[:,:,:,0]
print("data.shape", data.shape)

### Filter

## Moving Average --> Median
w=151
data = deMA(data, w)

## Threshold
t=1
data = Threshold(data, t)


## Morphological operations

# Erosion (2,2)
erosion_kernel = np.ones((2,2))
for t in range(data.shape[0]):
    data[t,:,:] = erosion(data[t,:,:], erosion_kernel)

## Select ROI only

## Visualize a chunk of the video

Watch(data)
