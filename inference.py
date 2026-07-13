import cv2
import numpy as np
import torch
from config import IMAGE_SIZE


def preprocess_image(image_path, device):
    """
    Reads and preprocesses an image for U-Net inference.

    Parameters
    ----------
    image_path : str
        Path to input image.
    device : torch.device
        CPU or CUDA device.
    image_size : tuple
        Resize dimensions.

    Returns
    -------
    image : numpy.ndarray
        Original RGB image.
    image_tensor : torch.Tensor
        Preprocessed tensor of shape (1, 3, H, W).
    """
    
    # Read image (BGR)
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(image, IMAGE_SIZE)

    # Normalize
    resized = resized.astype(np.float32) / 255.0

    # Convert to tensor
    image_tensor = torch.from_numpy(resized)

    # HWC -> CHW
    image_tensor = image_tensor.permute(2, 0, 1)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)

    return image, image_tensor


def predict_mask(model, image_tensor, threshold=0.5):
    """
    Performs segmentation using a trained U-Net.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net.
    image_tensor : torch.Tensor
        Tensor from preprocess_image().
    threshold : float
        Binary threshold.

    Returns
    -------
    predicted_mask : numpy.ndarray
        Binary mask (0 or 1).
    """
   
    model.eval()

    with torch.no_grad():

        prediction = model(image_tensor)

        prediction = torch.sigmoid(prediction)

        prediction = (prediction > threshold).float()

    predicted_mask = prediction.squeeze().cpu().numpy().astype(np.uint8)

    return predicted_mask


def resize_prediction(predicted_mask, original_image):
    """
    Resize predicted mask back to original image size.

    Parameters
    ----------
    predicted_mask : numpy.ndarray
        256×256 binary mask.
    original_image : numpy.ndarray
        Original RGB image.

    Returns
    -------
    resized_mask : numpy.ndarray
        Binary mask with original image dimensions.
    """
    
    resized_mask = cv2.resize(
        predicted_mask,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return resized_mask
