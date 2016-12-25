import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import getpass
import os
import errno
import numpy as np
import math
from matplotlib import pyplot as plt
from math import ceil, sqrt
import datetime
import sys
import random
import uuid
import logging
import cv2 as cv

def create_dir_if_necessary(path):
    """ Save way for creating a directory (if it does not exist yet). 
    From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary

    Keyword arguments:
    path -- path of the dir to check
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def get_uuid():
    """ Generates a unique string id."""

    x = uuid.uuid1()
    return str(x)
 
def get_parent_dir(path):
    """ Get the parent directory of a given path."""

    return os.path.abspath(os.path.join(path, os.pardir))

def check_if_dir_exists(path):
    """ Checks if a directory exists."""

    # From http://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python
    return os.path.isdir(path)

def check_if_file_exists(path):
    """ Cecks if a file exists."""

    return os.path.exists(path)

def remove_dir(path, warn=True):
    """ Removes a directory recursively. Prints a warning before if warn is True."""

    import shutil
    if warn:
        remove = radio_question("[!]", "Warning: Continue to delete path '{0}'? This will remove all files in this dir and all containing files and dirs.".format(path), None, ["Yes", "No"], [True, False])
    if not warn or remove:
        shutil.rmtree(path, True)

def increase_image_intesity(image, phi=1, theta=1):
    """ Changes the intensity of a given 8-bit grayscale image.

    Keyword arguments:
    phi -- makes the overall image darker. -- Default: 1
    theta -- increases brightness dark pixels. -- Default: 1
    """

    maxIntensity = 255.0

    # Increase intensity such that
    # dark pixels become much brighter, 
    # bright pixels become slightly bright
    newImage = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
    return np.array(newImage,dtype=np.uint8)

def decrease_image_intensity(image, phi=1, theta=1):
    """ Changes the intensity of a given 8-bit grayscale image.

    Keyword arguments:
    phi -- makes the overall image darker. -- Default: 1
    theta -- increases brightness dark pixels. Adds huge amounts of noise. -- Default: 1
    """

    maxIntensity = 255.0
    # Increase intensity such that
    # dark pixels become much brighter, 
    # bright pixels become slightly bright
    newImage = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
    return np.array(newImage,dtype=np.uint8)

def dilate_image(img, kernelSize=(4,4), iterations=2):
    """
    Performs an image dilation operation.
    
    Keyword arguments:
    kernelSize -- Size of the kernel to dilate the image. -- Default: (4,4)
    iterations -- Dilation iterations. -- Default: 2
    """

    kernel = np.ones(kernelSize, np.uint8)
    return cv.dilate(img,kernel, iterations=iterations) 

def opening_image(img, kernelSize=(4,4)):
    """
    Performs an image opening operation.
    
    Keyword arguments:
    kernelSize -- Size of the kernel to open the image. -- Default: (4,4)
    """

    kernel = np.ones(kernelSize, np.uint8)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

def calculate_point_distance(p1, p2):
    """ Calculates the distance between two points p1 and p2. """

    return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow(p1[1]-p2[1],2))

def crop_image(image, x, y, w, h):
    """ Crops an image.

    Keyword arguments:
    image -- image to crop
    x -- upper left x-coordinate
    y -- upper left y-coordinate
    w -- width of the cropping window
    h -- height of the cropping window
    """

    # do we need to convert the image?
    prevShape = image.shape
    image, reshaped = reshape_to_cv_format(image, False)

    # crop image using np slicing (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)
    image = image[y: y + h, x: x + w]
    if reshaped:        
        image = image.reshape(prevShape[0], image.shape[0], image.shape[1])
    return image

def crop_image_to_shape(image, x, y, shape):
    """ Wrapper for crop_image that encapsulates the cropping window as a tuple.

    Keyword arguments:
    image -- image to crop
    x -- upper left x-coordinate
    y -- upper left y-coordinate
    shape -- tuple (x, y) of the cropping window shape.
    """
    return crop_image(image, x, y, shape[0], shape[1])

def reduce_color(image):
    """ Reduces colors of an image."""

    # http://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv-in-python
    w, h, _ = image.shape
    for row in xrange(h-1):
        for col in xrange(w-1):
            #pi = row * w * 3 + col * 3
            pixel = image[col][row]
            pixel[0] = __reduceColorValue(pixel[0])
            pixel[1] = __reduceColorValue(pixel[1])
            pixel[2] = __reduceColorValue(pixel[2])
    return image

def __reduceColorValue(value):
    #if value < 64:
    #    return 0
    #if value < 128:
    #    return 64
    #return 255

    if value < 192:
       return int(value / 64.0  + 0.5) * 64
    return 255

def get_data_path():
    """ Returns the path to the current data directory."""
    return os.getcwd() + "/data/"

def get_temp_path():
    """ Returns the path to the current temporary directory."""
    return os.getcwd() + "/temp/"

def get_output_path():
    """ Returns the path to the current output directory."""
    return os.getcwd() + "/output/"

def get_today_string():
    """ Get todays date as a string representation. e.g.: 2016.4.18"""
    now = datetime.datetime.now()
    return "{0}.{1}.{2}".format(now.year, now.month, now.day)

def get_modified_median_cut_palette(image, colorCount=10, quality=1):
    """ 
    Calculates the modified median cut color palette.
    
    Keyword arguments:
    colorCount -- Number of colors to sample. -- Default: 10
    quality -- quality of sampling. -- Default: 1
    """

    w, h, _ = image.shape
    colors = []
    for row in xrange(0, h):
        for col in xrange(0, w, quality):
            pixel = image[col][row]
            b = pixel[0]
            g = pixel[1]
            r = pixel[2]
            if r < 250 and g < 250 and b < 250:
                colors.append((r, g, b)) 
    if len(colors) == 0:
        return [(255,255,255)]
    c_map = mmcq.mmcq(colors, colorCount) 
    return c_map.palette

def get_modified_median_cut_dominant_color(image, color_count=4, quality=1, palette=None):
    """ 
    Gets the dominant color for a given image as RGB Color.

    Keyword arguments:
    image -- the image to get the dominant color from
    color_count -- Number of colors to sample. -- Default: 4
    quality -- quality of sampling. -- Default: 1
    palette -- precomputed palette Default: None
    """
    if palette is None:
        palette = get_modified_median_cut_palette(image, color_count, quality)
    return palette[0]

def get_improved_median_cut(img):
    labels1 = segmentation.slic(img, compactness=30, n_segments=400)
    out1 = color.label2rgb(labels1, img, kind='avg')

    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg')
    
def show_image(image, windowTitle="image"):
    """ Shows an image using openCV and waits for the ESC key to be pressed."""

    cv.imshow(windowTitle, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_path_via_file_ui():
    """ Uses a file UI to get a path to a directory."""

    import Tkinter as tk
    import tkFileDialog as filedialog
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename() 

def display_images(images, imageConversion=cv.COLOR_BGR2RGB, titles=[], columns=4, rows=None, show=True):
    """
    Displays one or multiple images using matplotlib.

    Keyword arguments:
    images -- list of images
    imageConversion -- optional image Conversion. Note: OpenCV is BGR and matplotlib uses RGB. Use None for no image conversion.
    titles -- list of titles for the images
    columns -- number of image columns
    rows -- number of rows. Default: compute necessary number of rows
    show -- show result
    """
    if not show:
        return
    if imageConversion is not None:
        images = [cv.cvtColor(img, imageConversion) for img in images]

    # append filtered image
    if rows is None:
        rows = ceil(float(len(images)) / columns)

    try:
        for i in xrange(len(images)):
            plt.subplot(rows,columns,i+1),plt.imshow(images[i],'gray')
            if titles:
                plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
    except:
        logging.exception("Could not plot / show images. Saving instead.")
        save_plt_figure(plt, "img_show")

def save_plt_figure(figure, name, path=None):
    """ Save a matplotlib figure in the output directory."""

    if path is None:
        path = get_output_path() + get_today_string() + "/"
    try:
        create_dir_if_necessary(path)
        path += name + ".png"
        figure.savefig(path)
        return path
    except Exception as e:
        logging.exception("Could not save figure in {0}.".format(path))

def is_image_black(image, whitePixelThreshold=50, imageSum=None):
    """ 
    Decides if an image is "black".

    Keyword arguments:
    image -- image to test
    whitePixelThreshold -- number of white pixels threshold
    imageSum -- precomputed image sum
    """

    if image is None and imageSum is None:
        raise AttributeError("image and imageSum are None.")
    
    if imageSum is None:
        imageSum = np.sum(image)

    return imageSum < whitePixelThreshold * 255

def resize_image(image, resizeFactor):
    """ Resize image by a positive resizeFactor."""

    return cv.resize(image, (0,0), fx=resizeFactor, fy=resizeFactor)

def get_table(header, floatPercission=4, *rows):
    """
    Get ASCII table string.

    Keyword arguments:
    header -- table title
    floatPercission -- round precision for float values
    *rows -- data
    """

    table = PrettyTable(header)
    table.padding_width = 1
    for row in rows:
        # go through row and round floats
        for i in xrange(len(row)):
            if type(row[i]) is float:
                row[i] = round(row[i], floatPercission)
        table.add_row(row)
    return table




def annotate_points(ax, A, B):
    """ Annotate diagram points."""

    # from http://stackoverflow.com/questions/30039260/how-to-draw-cubic-spline-in-matplotlib
    for xy in zip(A, B):
        ax.annotate("{0}".format(round(xy[1], 3)), xy=xy, textcoords='offset points')

def show_progress(show, current, max, text, *args):
    """ 
    Shows the current process on the console.

    Keyword arguments:
    show -- Flag to display progress
    current -- current progress value
    max -- target progress value
    text -- optional text to print
    *args -- arguments for the text
    """
    if show:
        progress = round((float(current) / max) * 100.0, 0)
        output = "\r" + text.format(*args) + " {0}% done.       ".format(progress)                    
        sys.stdout.write(output)
        sys.stdout.flush() 

def choose_random_value(values, returnValue=True):
    """ Choses a random value from a list of values."""
    if returnValue:
        return values[random.randint(0, len(values)-1)]
    return random.randint(0, len(values)-1)

def clear_screen():
    """ Clears the screen on windows systems."""
    os.system('cls')

def flip_image_horizontal(image):
    """ Flips an image horizontally."""
    return cv.flip(image, 0)

def flip_image_vertical(image):
    """ Flips an image vertically."""
    return cv.flip(image, 1)

def flip_image(image, direction):
    """ Flips an image in a specified direction."""
    prevShape = image.shape
    image, reshaped = reshape_to_cv_format(image, False)
    image = cv.flip(image, direction)
    if reshaped:        
        image = image.reshape(prevShape)
    return image

def equalize_image_channel(channel):
    """ Histogram equalization of a single image channel."""

    if channel[0][0].shape == (3):
        raise AttributeError("More than one color channel.")
    return cv.equalizeHist(channel)

def ZCA_whitening(imageVector):
    """ ZCA whiten an image."""

    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

def equalize_BGR_image(image):
    """ Histogram eq whole color image."""

    b, g, r = cv.split(image)
    b = equalize_image_channel(b)
    g = equalize_image_channel(g)
    r = equalize_image_channel(r)
    return cv.merge((b,g,r))

def equalize_BGR_image_adaptive(image):
    """ Adaptive color image equalization (CLAHE)."""

    b, g, r = cv.split(image)
    b = equalize_image_channel_adaptive(b)
    g = equalize_image_channel_adaptive(g)
    r = equalize_image_channel_adaptive(r)
    return cv.merge((b,g,r))

def equalize_image_channel_adaptive(channel):
    """ Adaptive image channel equalization (CLAHE)."""
    if channel[0][0].shape == (3):
        raise AttributeError("More than one color channel.")
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(channel)

def change_light(image, value, channel="v"):
    """ Change the light intensity of an image."""

    channelDic = {"h": 0, "s":1, "v":2}
    # "translate" image channel to channel index
    if not channel in channelDic:
        raise AttributeError("invalid channel value. Valid values are h, s, or v")

    # which format (ConvNet (3, w, h) vs. Normal (w, h, 3)
    reshape = False
    prevShape = image.shape
    
    if image.shape[0] == 3 or image.shape[0] == 1:
        reshape = True
        if image.shape[0] == image.shape[1] or (image.shape[0] == 1 and image.shape[1] == 3): # grayscale 1L, 1L, h, w OR color 1L, 3L, h, w
            reshapeVector = (image.shape[2], image.shape[3], image.shape[1])         
        else:                      
            reshapeVector = (image.shape[1], image.shape[2], image.shape[0])                    # single row color or grayscale 1L/3L, h, w
        image = image.reshape(reshapeVector)
        
    #print "Shape",image.shape
    #print "dtype",image.dtype
    # convert to hsv
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # hsv[:,:,2] += value -  would be way faster but this does not prevent overflow (a high value gets even higher and becomes 0)
    channels = cv.split(hsv)
    for row in xrange(len(channels[channelDic[channel]])):
        for col in xrange(len(channels[channelDic[channel]][0])):
            channels[channelDic[channel]][row][col] = max(min(255, channels[channelDic[channel]][row][col]*value),0)

    image = cv.cvtColor(cv.merge(channels), cv.COLOR_HSV2BGR)

    # reshape back
    if reshape:        
        image = image.reshape(prevShape)
    return image

def change_brightness(image, value):
    """ Change the brightness intensity."""

    return change_light(image, value, "v")

def change_brightness_conv(image, value):
    """
    This function is called by the AugmentingBatchIterator.
    Because of the different scale of the images used for convnet training they have to be scaled back for opencv
    """
    image = rescale_image_0255(image)
    image = change_brightness(image, value)
    return rescale_image_01(image)

def change_saturation(image, value):
    """ Change saturation intensity."""
    return change_light(image, value, "s")

def change_saturation_conv(image, value):
    """
    This function is called by the AugmentingBatchIterator.
    Because of the different scale of the images used for convnet training they have to be scaled back for opencv
    """
    image = rescale_image_0255(image)
    image = change_saturation(image, value)
    return rescale_image_01(image)   
    
def rescale_image_01(image):
    """
    This contains the following operations
      1. Rescaling:               Images are scaled from [0, 255] to [0.0, 1.0]
    """
    # scale image to from [0, 255] to [0.0, 1.0]
    image = image.astype(np.float32)
    return image / 255 

def rescale_image_0255(image):
    """
    undo of rescale_image_01 operation.
    Needed for some OpenCV operations.
      1. Rescaling:               Images are scaled from [0.0, 1.0] to [0, 255]
    """
    # scale image to from [0.0, 1.0] to [0, 255]
    image *= 255
    return image.astype(np.uint8)

def reshape_to_cv_format(image, rescale=False):
    """ Reshape an image from the neural net format back to the OpenCV format."""
    reshape = False
    if rescale:
        image = rescale_image_0255(image)
    if image.shape[0] == 3 or image.shape[0] == 1:
        reshape = True
        if image.shape[0] == 1 and image.shape[1] == 3: # color 1L, 3L, h, w
            reshapeVector = (image.shape[2], image.shape[3], image.shape[1])         
        elif image.shape[0] == 3: # single row color 3L, h, w                      
            reshapeVector = (image.shape[1], image.shape[2], image.shape[0])                    
        elif image.shape[0] == 1 and image.shape[1] == 1: # grayscale 1L, 1L, h, w                         
            reshapeVector = (image.shape[2], image.shape[3])
        else: # single row grayscale 1L, h, w
            reshapeVector = (image.shape[1], image.shape[2])
        image = image.reshape(reshapeVector)
    return image, reshape

def rotate_image(image, angle, crop=True, keepSize=True):
    """ 
    Rotates an image by a free angle.

    Keyword arguments:
    image -- image to rotate
    angle -- angle to rotate in degrees.
    crop -- crop the resulting image?
    keepSize -- if True and the size has changed the image will be resized to fit the input size again
    """

     # do we need to convert the image?
    prevShape = image.shape
    image, reshaped = reshape_to_cv_format(image, False)

    # this is 10-times faster than: return ndimage.rotate(image, angle)
    # --
    # After http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    # Get the image size
    # No that's not an error - NumPy stores image matrices backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centered
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the transform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv.INTER_LINEAR
    )
    if not crop:
        # if we do not need to crop the image we can stop here, but the image might contain some black "corners"
        if reshaped:        
            result = result.reshape(prevShape[0], result.shape[0], result.shape[1])
        return result

    image = crop_around_center(result, *rotated_rect_with_max_area(image_size[0], image_size[1], math.radians(angle)))

    # if keepSize the same, the image is square and the size has changed than change size to fit input again
    if keepSize and image_size[0] == image_size[1] and (image_size[0] != image.shape[1] or image_size[1] != image.shape[0]):
        image = equalize_image_size(image, (image_size[0]*image_size[0]))

    if reshaped:        
        image = image.reshape(prevShape[0], image.shape[0], image.shape[1])
    return image

def translate_image_vertical(image, translationFactor):
    """ Translates the image vertically by a factor."""

    # create vertical translation matrix
    tm = np.float32([[1, 0, 0],
                     [0, 1, translationFactor]])
    return translate_image(image, tm)

def translate_image_horizontal(image, translationFactor):
    """ Translates the image horizontally by a factor."""
    # create horizontal translation matrix
    tm = np.float32([[1, 0, translationFactor],
                     [0, 1, 0]])
    return translate_image(image, tm)

def translate_image(image, translationMatrix):
    """ Translates the image given a translation matrix."""

    # which image shape? (ConvNet (3, w, h) vs. Normal (w, h, 3)
    reshape = False
    prevShape = image.shape
    if image.shape[0] == 3 or image.shape[0] == 1:
        reshape = True
        if image.shape[0] == image.shape[1] or (image.shape[0] == 1 and image.shape[1] == 3): # grayscale 1L, 1L, h, w OR color 1L, 3L, h, w
            reshapeVector = (image.shape[2], image.shape[3], image.shape[1])         
        else:                      
            reshapeVector = (image.shape[1], image.shape[2], image.shape[0])                    # single row color or grayscale 1L/3L, h, w
        image = image.reshape(reshapeVector)

    h, w = image.shape[0], image.shape[1]
    image = cv.warpAffine(image, translationMatrix, (w, h))

    if reshape:        
        image = image.reshape(prevShape)
    return image

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height around it's center point.
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    From http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr

def crop_to_square(image):
    """ Crops the square window of an iamge around the center."""

    if image is None:
        return None
    w, h = (image.shape[1], image.shape[0])
    w = float(w)
    h = float(h)

    # only crop images automatically if the aspect ratio is not bigger than 2 or not smaller than 0.5
    aspectRatio = w / h
    if aspectRatio > 3 or aspectRatio < 0.3:
        return None
    if aspectRatio == 1.0:
        return image
    
    # the shortest edge is the edge of our new square. b is the other edge
    a = min(w, h)
    b = max(w, h)

    # get cropping position
    x = (b - a) / 2.0

    # depending which side is longer we have to adjust the points
    # Heigth is longer
    if h > w:
        upperLeft = (0, x)        
    else:
        upperLeft = (x, 0)
    cropW = cropH = a    
    return crop_image(image, upperLeft[0], upperLeft[1], cropW, cropH)

def equalize_image_size(image, size):
    """ Resizes the image to fit the given size."""

    if image is None:
        return None

    # do we need to convert the image?
    prevShape = image.shape
    image, reshaped = reshape_to_cv_format(image, False)
    # image size
    w, h = (image.shape[1], image.shape[0])
        
    if (w*h) != size:
        # calculate factor so that we can resize the image. The total size of the image (w*h) should be ~ size.
        # w * x * h * x = size
        ls = float(h * w)
        ls = float(size) / ls
        factor = sqrt(ls)
        image = resize_image(image, factor)
    if reshaped:        
        image = image.reshape(prevShape[0], image.shape[0], image.shape[1])
    return image

def to_float32(n):
    """ Casts the input to a float32."""
    return np.cast["float32"](n)

def flatten_matrix(matrix):
    """ Flattens a np matrix."""

    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def visulize_matches(matches, k2, k1, img2, img1):
    """ Visualize SIFT keypoint matches."""

    import scipy as sp
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, :] = img1  
    view[:h2, w1:, :] = img2
    view[:, :, 1] = view[:, :, 0]  
    view[:, :, 2] = view[:, :, 0]

    for m in matches:
        m = m[0]
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        pt1 = (int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1]))
        pt2 = (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1]))

        cv.line(view, pt1, pt2, color)
    return view

