# note_for_DA3
My notes for Depth-Anything-3

## Reference
- Paper : [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
- Github : [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

## Introduction
- DA3 significantly outperforms **DA2** for monocular depth estimation, and **VGGT** for multi-view depth estimation and pose estimation.
- **DA3 Metric Series**(`DA3Metric-Large`) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

## System setup
### Package installation
```commandline
pip install torch\>2=2 torchvision
```