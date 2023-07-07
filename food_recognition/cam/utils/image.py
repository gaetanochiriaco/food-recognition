import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import List, Dict
from torchvision import transforms
from food_recognition.transforms import inv_normalization
from skimage import measure

def from_tensor_to_image(tensor,
                        num_element = 0,
                        inv_norm = True,
                        mean_img = [0.5457954, 0.44430383, 0.34424934],
                        sd_img = [0.23273608, 0.24383051, 0.24237761]):
    
    if type(tensor) == list:
        tensor = tensor[0]
    if inv_norm:
        t1 = inv_normalization(mean_img,sd_img)
        tensor = t1(tensor)
      

    if len(tensor.shape) == 4:
        img = tensor[num_element]
    else:
        img = tensor

    if img.shape[2] != 3 and img.shape[2] != 1:
        img = img.permute(1,2,0)


    img = torch.clip(img,min=0,max=1)
    img = img.detach().cpu().numpy()

    return img
    



def get_cam_on_image(img,
                      mask,
                      num_element= 0,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5,
                      mean_img = [0.5457954, 0.44430383, 0.34424934],
                        sd_img = [0.23273608, 0.24383051, 0.24237761]) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """

    if torch.is_tensor(img):
        img = from_tensor_to_image(img,num_element=num_element,mean_img=mean_img,sd_img=sd_img)
        
    
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    if len(mask.shape) == 4:
        mask = mask[num_element,:,:,:]
    

    heatmap = cv2.applyColorMap(np.uint8(255.0 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255.0 * cam)



def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result



def calculate_bbox(rows, cols):
    top = np.min(rows)
    bottom = np.max(rows)
    left = np.min(cols)
    right = np.max(cols)
    x = np.argmax((bottom-top,right-left))
    if x == 0:
      midpoint = (bottom+top)//2
      top = midpoint - (right-left)//2
      bottom = midpoint + (right-left)//2
      return [left, top, right, bottom]
    else: 
      midpoint = (right+left)//2
      left = midpoint - (bottom-top)//2
      right = midpoint + (bottom-top)//2
      return [left, top, right, bottom]
    



def return_bbox(cam_img, ratio):
    blobs = cam_img > ratio * np.max(cam_img)
    
    blobs_labels, blobs_num = measure.label(
        blobs, background=0, return_num=True)

    sum_label = {}
    for label in range(1, blobs_num + 1):
        current_sum = np.sum(cam_img[np.where(blobs_labels == label)])
        sum_label[current_sum] = label


    _,rows, cols = np.where(blobs_labels == sum_label[max(sum_label)])
    bbox = calculate_bbox(rows, cols)

    return bbox
