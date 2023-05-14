import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import List, Dict
from torchvision import transforms
from foodrecognition.transforms import inv_normalization
from skimage import measure



def show_cam_on_image(img,
                      mask,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5,
                      mean_img = [0.5457954, 0.44430383, 0.34424934],
                        sd_img = [0.23273608, 0.24383051, 0.24237761] ) -> np.ndarray:
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
        img = img.detach().cpu().numpy()
    
    if torch.is_tensor(mask):
        img = img.detach().cpu().numpy()

    if np.max(img) > 1 and np.min(img) < 0:
        t1 = transforms.ToTensor()
        t2 = inv_normalization(mean_img,sd_img)
        img = t1(img)
        img = t2(img)
        img = torch.clip(img,min=0,max=1)
        img = img.detach().cpu().numpy()

    if np.max(img) > 1:
        img = img/255
    

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def create_labels_legend(concept_scores: np.ndarray,
                         labels: Dict[int, str],
                         top_k=2):
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{','.join(labels[category].split(',')[:3])}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk


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


def scale_accross_batch_and_channels(tensor, target_size):
    batch_size, channel_size = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(
        batch_size * channel_size, *tensor.shape[2:])
    result = scale_cam_image(reshaped_tensor, target_size)
    result = result.reshape(
        batch_size,
        channel_size,
        target_size[1],
        target_size[0])
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
    



def return_bbox_and_region_cam(cam_img, ratio):
    blobs = cam_img > ratio * np.max(cam_img)
    #print("blobs: ", blobs)
    blobs_labels, blobs_num = measure.label(
        blobs, background=0, return_num=True)
    #print("blobs_num: ", blobs_num)

    sum_label = {}
    for label in range(1, blobs_num + 1):
        # print("label: ", label)
        current_sum = np.sum(cam_img[np.where(blobs_labels == label)])
        sum_label[current_sum] = label

    #print("sum_label: ", sum_label)
    #print("max(sum_label): ", max(sum_label))
    #print(sum_label[max(sum_label)])
    #print(blobs_labels.shape)
    _,rows, cols = np.where(blobs_labels == sum_label[max(sum_label)])
    bbox = calculate_bbox(rows, cols)
    region = (rows, cols)

    return bbox, region
