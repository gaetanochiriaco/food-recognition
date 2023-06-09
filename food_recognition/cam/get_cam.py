import torch
from food_recognition.cam.eigen_cam import EigenCAM

def get_output_and_cam(model: torch.nn.Module,
                       target_layer: torch.nn.Module,
                       reshape_transform,
                       input_tensor: torch.nn.Module,
                       method=EigenCAM,
                       aug_smooth = False,
                       eigen_smooth = False):
    
        cam = method(model=model,
                      target_layers=target_layer,
                      use_cuda = True,
                      reshape_transform=reshape_transform
               )

        output, batch_results = cam(input_tensor=input_tensor,
                                    targets = None,
                                    aug_smooth = aug_smooth,
                                    eigen_smooth = eigen_smooth)
        

        return output, batch_results



