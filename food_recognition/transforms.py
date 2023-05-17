from torchvision import transforms
from timm.data.transforms import  RandomResizedCropAndInterpolation
import random
from PIL import ImageFilter, ImageOps

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        


def get_transforms_list(image_size = (224,224),
                        mean_img = [0.,0.,0.],
                        sd_img = [1.,1.,1.],
                        rrci = False,
                        h_flip = False,
                        aug3 = False,
                        color_jitter = False):
    list_aug = []

    if rrci:
        list_aug.append(
            RandomResizedCropAndInterpolation(image_size,scale=(0.08, 1.0),  interpolation="bicubic")
        )

    else:
        list_aug.append(transforms.Resize(image_size))


    if h_flip:
        list_aug.append(
            transforms.RandomHorizontalFlip(),
         )
        

    if aug3:
         list_aug.append(
         transforms.RandomChoice([transforms.Grayscale(3),
                             Solarization(p=1.0),
                             GaussianBlur(p=1.0)])
         )
    
    if color_jitter:
        list_aug.append(
             transforms.ColorJitter(0.3,0.3,0.3)
        )
    
    list_aug.append(transforms.ToTensor())
    list_aug.append(transforms.Normalize(mean_img,sd_img))

    transform_train = transforms.Compose(list_aug)

    # transforms of test dataset
    transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, sd_img),

    ])
    
    return transform_train,transform_test


def inv_normalization(mean_img,std_img):

    new_mean = [0.,0.,0.]
    new_std = [1.,1.,1.]

    for i in range(3):
        new_mean[i] = -(mean_img[i]/std_img[i])
        new_std[i] = (1/std_img[i])
    
    return transforms.Normalize(new_mean,new_std)


def get_inv_transform_list(mean_img,
                           std_img,
                           to_pil = True):
    transforms_list = [inv_normalization(mean_img,std_img)]

    if to_pil:
        transforms_list.append(transforms.ToPILImage())
    

    return transforms.Compose(transforms_list)
    
