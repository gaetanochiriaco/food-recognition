import matplotlib.pyplot as plt
import torch 
import torchvision.transforms as T
from food_recognition.cam.utils.image import from_tensor_to_image

def plot_bbox_on_image(img,bbox,save=True):
    

    img = from_tensor_to_image(img)
    
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


def plot_image(img,num_element=0,save=True):


    img = from_tensor_to_image(img,num_element=0)
    
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Need to implement
def plot_batch(img_batch, plot_fn,num_print=None,**kwargs):

    if num_print is None:
        num_print 

    if img_batch.shape[0] >16:
        print("Batch size is too high to print!")
#     h, w = 10, 10        # for raster image
#     nrows, ncols = 5, 4  # array of sub-plots
#     figsize = [6, 8]     # figure size, inches

#     # prep (x,y) for extra plotting on selected sub-plots
#     xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
#     ys = np.abs(np.sin(xs))           # absolute of sine

# # create figure (fig), and array of axes (ax)
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

# # plot simple raster image on each sub-plot
# for i, axi in enumerate(ax.flat):
#     # i runs from 0 to (nrows*ncols-1)
#     # axi is equivalent with ax[rowid][colid]
#     img = np.random.randint(10, size=(h,w))
#     axi.imshow(img, alpha=0.25)
#     # get indices of row/column
#     rowid = i // ncols
#     colid = i % ncols
#     # write row/col indices as axes' title for identification
#     axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))

# # one can access the axes by ax[row_id][col_id]
# # do additional plotting on ax[row_id][col_id] of your choice
# ax[0][2].plot(xs, 3*ys, color='red', linewidth=3)
# ax[4][3].plot(ys**2, xs, color='green', linewidth=3)

# plt.tight_layout(True)
# plt.show()



