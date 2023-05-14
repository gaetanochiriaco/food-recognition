import torch
from pytorch_grad_cam import EigenCAM

def get_output_and_cam(model: torch.nn.Module,
                       target_layer: torch.nn.Module,
                       reshape_transform,
                       input_tensor: torch.nn.Module,
                       method=EigenCAM):
    
        cam = method(model=model,
                      target_layers=target_layer,
                      use_cuda = True,
                      reshape_transform=reshape_transform
               )

        output, batch_results = cam(input_tensor=input_tensor,
                                    targets = None,
                                    aug_smooth = False,
                                    eigen_smooth = False)
        

        return output, batch_results



