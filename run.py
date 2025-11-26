import glob, os, torch
from depth_anything_3.api import DepthAnything3

def da3_model_initial():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained('depth-anything/da3metric-large')
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Initialize the DA3 model
    model = da3_model_initial()
    