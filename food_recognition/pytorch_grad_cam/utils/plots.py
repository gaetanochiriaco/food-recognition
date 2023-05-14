import matplotlib.pyplot as plt
import torch 
import torchvision.transforms as T

def plot_bbox_on_image(img,bbox,save=True):
    
    if torch.is_tensor(img):
        transform = T.ToPILImage()
        img = transform(img)
    
    plt.figure(figsize=(5,5))
    fig, ax = plt.subplots()
    ax.imshow(img)
    x0, y0 = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    plt.axis('off')

    if save:
        plt.savefig("bbox.jpg",bbox_inches='tight',pad_inches=0)
    plt.show()  


