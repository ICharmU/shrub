from skimage.transform import rescale, resize
import numpy as np

def scale_resolution_2d(img: np.ndarray, scale_factor: float, height: int, width: int):
    """
    Scales an image by a specified factor and distorts the image to the specified dimensions.

    Args:
        img: 2d np array. Expects the color channel to be the last argument. If using grayscale images, be sure that the last dimension has size 1.
        scale_factor: factor by which the scale the height and width of the image.
        height, width: dimensions of the output image

    Returns:
        rescaled: 2d np array with dimensions (height, width, color channels)
    """
    scale = [scale_factor for _ in range(len(img.shape))]
    scale[-1] = 1 # don't scale last dim. this changes the color channels used

    rescaled = rescale(img, scale)
    rescaled = resize(rescaled, (height,width))
    return rescaled

def full_image_accuracy(high_res_img: np.ndarray, low_res_img: np.ndarray, gamma: float = 1/255):
    """
    Expects a high resolution image (larger dimensions) and low resolution image (smaller dimensions) of the same size. 
    Compares all pixels within the image and checks if their average color scale difference is within gamma.

    Args:
        high_res_img, low_res_img - same size 2d arrays to compare values of. Both images should be normalized to [0,1]
        gamma - used as a margin of error. By default gamma is set to 1/255 in expectation of a [0,1] normalized color scale.

    Returns:
        combined_accuracy - accuracy of all pixels, regardless of class
        scale_match_masked -  array displaying predictions sufficiently close to the label. 
                             a filler of 1 is used for incorrect predictions.
                             values are renormalized which causes increased contrast compared to the input images.
    """
    eps = 10**-12
    scale_match = high_res_img - low_res_img
    mask = np.where(np.abs(scale_match).mean(axis=-1) < gamma) # average absolute difference
    scale_match_masked = np.ones_like(scale_match) # zero difference indicates equality
    scale_match_masked[mask] = (scale_match[mask] - scale_match[mask].min()) / (scale_match[mask].max() - scale_match[mask].min() + eps) # keep interval [0,1]

    combined_accuracy = (scale_match_masked != 1).mean()

    return combined_accuracy, scale_match_masked

def shrub_only_accuracy(label_img: np.ndarray, preds: list):
    """
    Finds the accuracy of individual shrub predictions. For full image predictions use full_image_accuracy_2d() instead for vectorized operations.

    Args:
        label_img - np array representing 2d binary image.
        preds - list containing elements of the form (bool pred, *coords) where coords is typically (x,y) or (x,y,z)

    Ex. (2d)
    samp_img = np.array([
        [1,0],
        [0,0]
    ])
    samp_preds = [(1,0,0), (0,0,1), (1,1,1)]

    shrub_only_accuracy_2d(samp_img, samp_preds)

    Output:
    0.6666666666666666

    Ex. (3d)
    samp_img = np.array([
        [[1,0,0],
        [0,0,1]],
        [[1,1,0],
        [0,1,1]],
    ])
    samp_preds = [(1,0,0,0), (0,0,1,0), (1,1,1,0), (0,1,0,1), (0,0,1,1)]

    shrub_only_accuracy_2d(samp_img, samp_preds)

    Output:
    0.6
    """
    if len(preds) == 0:
        return None
    
    n_correct = 0
    for pred, *coords in preds:
        n_correct += pred == label_img[*coords]
    
    accuracy = n_correct / len(preds)
    return accuracy