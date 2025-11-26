import glob, os, torch
from depth_anything_3.api import DepthAnything3

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained('depth-anything/da3metric-large')
    print(model)


if __name__ == '__main__':
    main()