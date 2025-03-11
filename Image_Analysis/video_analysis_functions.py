import numpy as np
import shelve
import os, signal

import plotly.express as px
import plotly.graph_objects as go
import json
import dash
from dash import Dash, dcc, html, Input, Output, State
from flask import request

from skimage import io, draw, measure
from skimage.filters import threshold_otsu
from skimage.morphology import (erosion, skeletonize)
from sympy import python
from scipy import ndimage

#### Filter functions

### GUI

## ROI drawing from an image (or even better, a video)

def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)

def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def ROI(data, ROI_filename = ""):

    ## Draw or Import ROI, depending if a filename is given or not
    if ROI_filename == "":

        ROI_filename = 'Output\Shelves\shelve_new.out'
        my_shelf = shelve.open(ROI_filename,'n') # New shelf

        app = Dash(__name__)

        fig = px.imshow(data, animation_frame=0)
        fig.update_layout(dragmode='drawclosedpath', newshape=dict(line_color='cyan'))

        fig_mask = px.imshow(data, animation_frame=0)

        app.layout = html.Div([
        dcc.Graph(
            id='graph', 
            config={'modeBarButtonsToAdd':['drawclosedpath', 'eraseshape']}, 
            figure=fig,
            style={'width':600, 'height':600, 'margin-left':'auto', 'margin-right':'auto'}
        ),
        html.Button(id='apply-button-state', n_clicks=0, children='Apply mask'),
        dcc.Graph(id='my-output', 
        figure=fig_mask, 
        style={'width':600, 'height':600, 'margin-left':'auto', 'margin-right':'auto'}
        ),
        html.Button(id='save-button-state', n_clicks_timestamp=-1, children='Save mask')
        ])

        @app.callback(
            Output('my-output', 'figure'),
            Input('save-button-state', 'n_clicks_timestamp'),
            Input('apply-button-state', 'n_clicks'),
            State('graph', 'relayoutData'),
            prevent_initial_call=True,
        )

        def on_new_annotation(n_clicks_timestamp_save, n_clicks_apply, relayout_data):

            img = np.copy(data)
            if "shapes" in relayout_data or "shapes[0].path" in relayout_data:
                if "shapes" in relayout_data:    
                
                    drawn_shape = relayout_data["shapes"][-1]
                    mask = path_to_mask(drawn_shape["path"], img[0].shape)
                
                else:
                    modified_shape = relayout_data["shapes[0].path"]
                    mask = path_to_mask(modified_shape, img[0].shape)

                if n_clicks_timestamp_save > -1: # When clicked on save for the first time
                    my_shelf['mask'] = locals()['mask']
                    my_shelf.close()

                for t in range(img.shape[0]):
                    img[t][mask]=0    
                
                return px.imshow(img, animation_frame=0)

            else:
                return dash.no_update

        app.run_server(debug=False)

    else:

        my_shelf = shelve.open(ROI_filename) # Open shelf
        ROI_mask = my_shelf['mask']
        my_shelf.close()
    
    ## Apply ROI mask to data

    for t in range(data.shape[0]):
        data[t][ROI_mask]=0

    return data

### Moving average

def MA(data, w):
    new_data = np.zeros((data.shape[0]-w+1, data.shape[1], data.shape[2]))
    # Going through pixels
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            new_data[:,i,j] = np.convolve(data[:,i,j], np.ones(w), 'valid') / w
    return new_data

### de-Moving Average

def deMA(data, w):
    new_data = np.zeros((data.shape[0]-w+1, data.shape[1], data.shape[2]))
    # Going through pixels
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            new_data[:,i,j] = data[int((w-1)/2):-int((w-1)/2),i,j] - np.convolve(data[:,i,j], np.ones(w), 'valid') / w
    return new_data

### Binary projection with a threshold

def Threshold(data, t):
    binary_data = np.ones(data.shape)
    thresh = threshold_otsu(data)
    binary_data = data < thresh
    return binary_data

### Threshold on cluster size after cluster labelling

def ClusterThreshold(data, ct):
    for t in range(data.shape[0]):
        labels = measure.label(data[t])
        for region in measure.regionprops(labels):
            if region.area <= ct:
                small_cluster_points = np.where(labels == region.label)
                data[t][small_cluster_points] = 0
    return data

### Skeletonization

def Skeletonize(data):
    for t in range(data.shape[0]):
        data[t] = skeletonize(data[t])
    return data

#### Analysis and data extraction

def Extract_Axonemes(image):
    """Extract the two axonemes from one image, as un unordered list of coordinates."""

    # Unordered list of points
    NoN = np.nonzero(image)

    # Rearranging as a list of coordinates
    axonemes_disordered = []
    for k in range(len(NoN[0])):
       axonemes_disordered.append([NoN[1][k],NoN[0][k]])

    return axonemes_disordered

def ExtractAxonemes(data):
    """ Extract from a video the axonemes at each timeframe, 
    in the form of a list of n_t x 2 arrays, where n_t can vary """

    axonemes_disordered_video = []
    for t in range(data.shape[0]):
        NoN = np.nonzero(data[t])

        axonemes_disordered = np.zeros((len(NoN[0]),2))
        for k in range(axonemes_disordered.shape[0]):
            axonemes_disordered[k][0] = NoN[1][k]
            axonemes_disordered[k][1] = NoN[0][k]
        
        axonemes_disordered_video.append(axonemes_disordered)
    return axonemes_disordered_video


def CenterOfMass(data):
    M = measure.moments(data[0], order=1)
    centroid = [0, 0]
    for t in range(data.shape[0]):
        M = measure.moments(data[t], order=1)
        centroid[0] += M[1, 0] / M[0, 0]
        centroid[1] += M[0, 1] / M[0, 0]
    centroid[0] = centroid[0] / data.shape[0]
    centroid[1] = centroid[1] / data.shape[0]
    return centroid

def basal_end(coordinates, point, orientation="none", orientation_threshold=-1):
    """ Identify, in a unordered list of coordinates (n_t arrays elements of dimension 2) the closest point to a given point,
    either left or right-located, or without orientation.
    Applied this code to 
    - identify the basal end from center of mass coordinates.
    - use it recursively to order the unordered list of points of both axonemes.
    If orientation is specified, one will look at the basal end on the left side or the right side of the point. """

    res_coord_index = -1
    min_dist = 126**2
    for k in range(len(coordinates)):
        
        dist = min_dist+1
        if (orientation=="left" and orientation_threshold>-1 and coordinates[k][0]<orientation_threshold) or (orientation=="right" and orientation_threshold>-1 and coordinates[k][0]>orientation_threshold) or (orientation=="none" and orientation_threshold==-1):
                dist = (coordinates[k][0]-point[0])**2 + (coordinates[k][1]-point[1])**2
                # print("coordinates[k]", coordinates[k], "point", point, "orientation threshold", orientation_threshold)     

        if dist<min_dist:
            res_coord_index = k
            min_dist = dist
    
    # Deletion of minimum from coordinates and returning it at the same time
    if res_coord_index<0:
        return None
    else:
        return coordinates.pop(res_coord_index)

def OrderAxonemes(Disordered_axonemes, center_of_mass):

    Ordered_left_axonemes= []
    Ordered_right_axonemes = []    

    for t in range(len(Disordered_axonemes)):

        unordered_coordinates = np.copy(Disordered_axonemes[t]).tolist() #can put out deep copy when debugged

        # Ordered_left_axoneme, Ordered_right_axoneme = Order_axonemes(Disordered_axonemes[t], center_of_mass)

        axoneme_left = []
        axoneme_right = []

        # print("length of unordered list: ", len(unordered_coordinates))
        # Determine basal ends from point and store it in two newly created lists
        # print("Basal end, left axoneme:")
        bl = basal_end(unordered_coordinates, center_of_mass, 'left', center_of_mass[0])
        if bl!=None:
            axoneme_left.append(bl)
        br = basal_end(unordered_coordinates, center_of_mass, 'right', center_of_mass[0])
        if t==84:
            print("bl: ", bl)
            print("br: ", br)
            print("unordered coordinates: ", unordered_coordinates)
        if br!=None:
            axoneme_right.append(br)
        # print("Basal end, right axoneme:")

        print("axoneme basal ends (t="+str(t)+"): ", axoneme_left, axoneme_right)

        # Recursively fill ordered axonemes with same function
        while len(unordered_coordinates)>0:
            # print("new basal end, left axoneme: ")
            bl=basal_end(unordered_coordinates, axoneme_left[-1], 'left', axoneme_right[0][0])
            if bl!=None:
                axoneme_left.append(bl)
            if len(unordered_coordinates)>0:
                # print("new basal end, right axoneme: ")
                br=basal_end(unordered_coordinates, axoneme_right[-1], 'right', axoneme_left[0][0])
                if br!=None:
                    axoneme_right.append(br)


        Ordered_left_axonemes.append(np.array(axoneme_left))
        Ordered_right_axonemes.append(np.array(axoneme_right)) 


    return Ordered_left_axonemes, Ordered_right_axonemes

def ExtractAndOrderAxonemes(data):
    """ Extracts an unordered list of points representing the two axonemes,
            Computes the center of mass between the two axonemes,
            Orders and separates the two axonemes from the basal end to the distal end."""

    Disordered_axonemes = ExtractAxonemes(data)
    center_of_mass = CenterOfMass(data)
    print("Center of mass in ExtractAndOrder(data): ", center_of_mass)

    Ordered_left_axonemes, Ordered_right_axonemes = OrderAxonemes(Disordered_axonemes, center_of_mass)

    return Ordered_left_axonemes, Ordered_right_axonemes 

#### Visualisations

def Watch(data):
    img = data[100:400]
    fig = px.imshow(img, animation_frame=0, labels=dict(animation_frame="slice"))
    fig.show() 
    return

#### Tests

if __name__=="__main__":

    ### Open a tif-stack file

    original_data = io.imread('C:\\Users\\Luc\\Documents\\MEGAsync\\PhD\\RheoFlag\\Code\\Image_Analysis\\Input_Stack.tif')

    ### Prepare the data

    ## Convert into grayscale
    data = original_data[:,:,:,0]
    print("data.shape", data.shape)

    ## Visualize a chunk of the video

    # Watch(data)

    ### Filter
    Filter = True
    if Filter:

        ## deMoving Average
        w=151
        data = deMA(data, w)
        print("After deMa: data.shape = ", data.shape)

        ## Threshold
        t=1
        data = Threshold(data, t)
        print("Threshold done")

        ## Morphological operations

        # Erosion (2,2)
        erosion_kernel = np.ones((2,2))
        for t in range(data.shape[0]):
            data[t,:,:] = erosion(data[t,:,:], erosion_kernel) 
        print("Erosion done")

        ### ROI masking
        filename = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Image_Analysis\\Output\\Shelves\shelve.out"
        data = ROI(data, ROI_filename=filename)
        # data = ROI(data) # Use that option to draw ROI
        print("ROI done.")

        ### Cluster thresholding
        cluster_threshold = 10
        data = ClusterThreshold(data, cluster_threshold)
        print("Cluster thresholding done")

        ### Skeletonization
        data = Skeletonize(data)
        print("Skeletonization done")

        ### Reclustering on smaller elements (due to thinning)
        cluster_threshold = 15
        data = ClusterThreshold(data, cluster_threshold)
        # Watch(data)
        print("Reclustering done")

        #### Data extraction and Analysis

        # Disordered_axonemes = ExtractAxonemes(data)

        # fig = px.imshow(data[83])
        # fig.add_trace(go.Scatter(x = Disordered_axonemes[83][:,0], y = Disordered_axonemes[83][:,1], mode="markers", marker=dict(size=5, color="MediumPurple", opacity=0.6)))
        # fig.show()
        # exit()

        Ordered_left_axonemes, Ordered_right_axonemes = ExtractAndOrderAxonemes(data)
        print("Extraction and Ordering of axonemes from skeleton: done")

        Ordered_left_axoneme = Ordered_left_axonemes[10]
        Ordered_right_axoneme = Ordered_right_axonemes[10]
    
        fig = px.imshow(data[10])
        fig.add_trace(go.Scatter(x = Ordered_left_axoneme[:,0], y = Ordered_left_axoneme[:,1], mode="lines+markers", marker=dict(size=5, color="MediumPurple", opacity=0.6)))
        fig.add_trace(go.Scatter(x = Ordered_right_axoneme[:,0], y = Ordered_right_axoneme[:,1], mode="lines+markers", marker=dict(size=5, color="LightSeaGreen", opacity=0.6)))
        fig.show()


    ### Figure
    # Let's build an animation of image frames, first. Then we add traces such as scatter plots.

    ## data
    # Put data of the all animation
    # print("px.imshow(data[0]).data[0]: ")
    # print(px.imshow(data[0]).data[0])

    exit()
    fig = px.imshow(data[0])
    fig.add_trace(go.Scatter(x = Ordered_left_axonemes[0][:,0], y = Ordered_left_axonemes[0][:,1], mode="lines+markers", marker=dict(size=4, color="MediumPurple", opacity=0.8)))
 

    ## frames

    frames_animation = [dict(data = [px.imshow(data[t]).data[0], go.Scatter(x = Ordered_left_axonemes[t][:,0], y = Ordered_left_axonemes[t][:,1])] , name = str(t)) for t in range(data.shape[0])]

    Fig = go.Figure(data = fig.data, frames=frames_animation, layout = fig.layout)
    
    ## layout update
    Fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "animation_frame="},
                "len": 1.2,
                "steps": [
                    {
                        "args": [
                            [fr.name],
                            {
                                "frame": {"duration": 10, "redraw": True},
                                "mode": "immediate",
                                "fromcurrent": True,
                            },
                        ],
                        "label": fr.name,
                        "method": "animate",
                    }
                    for fr in Fig.frames
                ],
            }
        ],
        # updatemenus = [dict(type = "buttons", 
        #                             buttons = [dict(label = "Play", 
        #                                         method = "animate", 
        #                                         args = [None]), 
        #                                         dict(label = "Pause", 
        #                                         method = "animate", 
        #                                         args = ["Null"])]
        #                                     )]
    )

    Fig.show()