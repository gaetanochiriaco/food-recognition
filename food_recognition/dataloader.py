import torch
import PIL
import os
from food_recognition.transforms import get_transforms_list
from torchvision.datasets import Food101

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1].strip())))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.image_path = image_path

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img = self.loader(os.path.join(self.image_path, img_name))

        # print img
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def load_data(image_path, 
              train_dir, 
              test_dir, 
              batch_size,
              image_size=(224,224), 
              mean_img = [0.5457954, 0.44430383, 0.34424934],
              sd_img = [0.23273608, 0.24383051, 0.24237761],
              rrci = True,
              h_flip = True,
              aug3 = True,
              color_jitter = True):
    

    transform_train, transform_test = get_transforms_list(image_size,mean_img,sd_img,rrci,h_flip,aug3,color_jitter)

    train_dataset = MyDataset(txt_dir=train_dir, image_path=image_path, transform=transform_train)
    test_dataset = MyDataset(txt_dir=test_dir, image_path=image_path, transform=transform_test)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size,  shuffle=False, num_workers=2)
    return train_dataset, train_loader, test_dataset, test_loader



def load_data_101(image_path,
                  batch_size,
                  image_size = (224,224),
                  mean_img = [0.485, 0.456, 0.406],
                  sd_img = [0.229, 0.224, 0.225],
                  rrci = True,
                  h_flip = True,
                  aug3 = True,
                  color_jitter = True):
    
   transform_train, transform_test = get_transforms_list(image_size,mean_img,sd_img,rrci,h_flip,aug3,color_jitter)
   
   train_dataset = Food101(root=image_path, split="train", download=False,transform=transform_train)
   test_dataset = Food101(root=image_path, split="test", download=False,transform=transform_test)
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 2)
    
   return train_dataset, train_loader, test_dataset, test_loader