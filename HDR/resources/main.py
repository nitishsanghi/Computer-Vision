"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code for above & beyond effort, make sure
that you include important snippets in your writeup. CODE ALONE IS
NOT SUFFICIENT FOR ABOVE AND BEYOND CREDIT.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2
import numpy as np

import os
import errno

from os import path

import hdr as hdr


# Change the source folder and exposure times to match your own
# input images. Note that the response curve is calculated from
# a random sampling of the pixels in the image, so there may be
# variation in the output even for the example exposure stack
SRC_FOLDER = "images/source/livingroom"
#EXPOSURE_TIMES = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
#                             1 / 60.0, 1 / 40.0, 1 / 15.0])
EXPOSURE_TIMES =  np.float64([1/4000.0, 1/2000.0, 1/1000.0, 1/500.0,1/250.0, 1/125.0, 1/60.0, 1/30.0, 1/15.0, 1/8.0, 1/4.0])
#1/4000.0, 1/2000.0, 1/1000.0, 1/500.0,
OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def computeHDR(images, log_exposure_times, smoothing_lambda=100.):
    """Computational pipeline to produce the HDR images according to the
    process in the Debevec paper.

    NOTE: This function is NOT scored as part of this assignment.  You may
          modify it as you see fit.

    The basic overview is to do the following for each channel:

    1. Sample pixel intensities from random locations through the image stack
       to determine the camera response curve

    2. Compute response curves for each color channel

    3. Build image radiance map from response curves

    4. Apply tone mapping to fit the high dynamic range values into a limited
       range for a specific print or display medium (NOTE: we don't do this
       part except to normalize - but you're free to experiment.)

    Parameters
    ----------
    images : list<numpy.ndarray>
        A list containing an exposure stack of images

    log_exposure_times : numpy.ndarray
        The log exposure times for each image in the exposure stack

    smoothing_lambda : np.int (Optional)
        A constant value to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.

    Returns
    -------
    numpy.ndarray
        The resulting HDR with intensities scaled to fit uint8 range
    """
    images = [np.atleast_3d(i) for i in images]
    num_channels = images[0].shape[2]

    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):

        # Collect the current layer of each input image from
        # the exposure stack
        layer_stack = [img[:, :, channel] for img in images]

        # Sample image intensities
        intensity_samples = hdr.sampleIntensities(layer_stack)

        # Compute Response Curve
        response_curve = hdr.computeResponseCurve(intensity_samples,
                                                  log_exposure_times,
                                                  smoothing_lambda,
                                                  hdr.linearWeight)

        # Build radiance map
        img_rad_map = hdr.computeRadianceMap(layer_stack,
                                             log_exposure_times,
                                             response_curve,
                                             hdr.linearWeight)

        # We don't do tone mapping, but here is where it would happen. Some
        # methods work on each layer, others work on all the layers at once;
        # feel free to experiment.  If you implement tone mapping then the
        # tone mapping function MUST appear in your report to receive
        # credit.
        
        out = np.zeros(shape=img_rad_map.shape, dtype=img_rad_map.dtype)
        cv2.normalize(img_rad_map, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        hdr_image[..., channel] = out

    return hdr_image


def main(image_files, output_folder, exposure_times, resize=False):
    """Generate an HDR from the images in the source folder """

    # Print the information associated with each image -- use this
    # to verify that the correct exposure time is associated with each
    # image, or else you will get very poor results
    print("{:^30} {:>15}".format("Filename", "Exposure Time"))
    print("\n".join(["{:>30} {:^15.4f}".format(*v)
                     for v in zip(image_files, exposure_times)]))

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    if any([im is None for im in img_stack]):
        raise RuntimeError("One or more input files failed to load.")

    # Subsampling the images can reduce runtime for large files
    if resize:
        img_stack = [img[::4, ::4] for img in img_stack]

    log_exposure_times = np.log(exposure_times)
    hdr_image = computeHDR(img_stack, log_exposure_times)
    cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)
    
    
    ############ ABOVE AND BEYOND CODE ##### ABOVE AND BEYOND CODE ################################
    ############ ABOVE AND BEYOND CODE ##### ABOVE AND BEYOND CODE ################################
    ############ ABOVE AND BEYOND CODE ##### ABOVE AND BEYOND CODE ################################
    
    gamma = 2.2
    tone_map_image = (np.power(hdr_image/255.0,1.0/gamma)*255).astype(dtype='uint8')
    cv2.imwrite(path.join(output_folder, "outputtoned.png"), tone_map_image)
    tone_map_image = (np.power(hdr_image/255.0,gamma)*255).astype(dtype='uint8')
    cv2.imwrite(path.join(output_folder, "outputtonedlinear.png"), tone_map_image)
    hdr_image = hdr_image.astype(dtype='uint8')
    img_rad_map = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2LAB)
    cv2.imwrite(path.join(output_folder, "outputlab.png"), tone_map_image)
    img_rad_map[:,:,0] = (np.power(img_rad_map[:,:,0]/255.0,1.0/gamma)*255).astype(dtype='uint8')
    tone_map_image = cv2.cvtColor(img_rad_map, cv2.COLOR_LAB2BGR)
    cv2.imwrite(path.join(output_folder, "outputtonedlab.png"), tone_map_image)
    img_rad_map = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2LAB)
    lab_channels = cv2.split(img_rad_map)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    lab_channels[0] = clahe.apply(lab_channels[0])
    img_rad_map = cv2.merge(lab_channels)
    tone_map_image = cv2.cvtColor(img_rad_map, cv2.COLOR_LAB2BGR)
    cv2.imwrite(path.join(output_folder, "outputtonedclahe.png"), tone_map_image)
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
    print("Done!")


if __name__ == "__main__":
    """Generate an HDR image from the images in the SRC_FOLDER directory """

    np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = next(src_contents)

    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print("Processing '" + image_dir + "' folder...")

    image_files = sorted([os.path.join(dirpath, name) for name in fnames
                          if not name.startswith(".")])

    main(image_files, output_dir, EXPOSURE_TIMES, resize=False)
