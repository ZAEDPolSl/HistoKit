
def gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian blur to the input image.

    Parameters:
    - image: Input image as a NumPy array.
    - kernel_size: Size of the Gaussian kernel.
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - Blurred image as a NumPy array.
    """
    import cv2
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def motion_blur(image, kernel_size=15, angle=0):
    """
    Apply motion blur to the input image.

    Parameters:
    - image: Input image as a NumPy array.
    - kernel_size: Size of the motion blur kernel.
    - angle: Angle of motion blur in degrees.

    Returns:
    - Blurred image as a NumPy array.
    """
    import cv2
    import numpy as np

    # Create the motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0), (kernel_size, kernel_size))
    kernel = kernel / kernel_size

    # Apply the kernel to the image
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def median_blur(image, kernel_size=5):
    """
    Apply median blur to the input image.

    Parameters:
    - image: Input image as a NumPy array.
    - kernel_size: Size of the median blur kernel.

    Returns:
    - Blurred image as a NumPy array.
    """
    import cv2
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image