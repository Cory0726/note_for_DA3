# note_for_DA3
My notes for Depth-Anything-3

- [Reference](#reference)
- [Introduction](#introduction)
- [System setup](#system-setup)
  - [Python version](#python-version--39--313)
  - [Package](#package-installation)
  - [Issue](#issue)

## Reference
- Paper : [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
- Github : [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

## Introduction
- DA3 significantly outperforms **DA2** for monocular depth estimation, and **VGGT** for multi-view depth estimation and pose estimation.
- **DA3 Metric Series**(`DA3Metric-Large`) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

## System setup
### Python version : `>=3.9 , <=3.13`
### Package installation
```
# torch, torchvision
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126

# xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126

# depth-anything-3
pip install -e .

# scikit-learn
pip install scikit-learn
```
### Issue
#### Enable long paths in Windows 10, version 1607, and later
- Windows official web : [Enable long paths in Windows 10, version 1607, and later](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#enable-long-paths-in-windows-10-version-1607-and-later)
- Solution :
```commandline
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
#### ModuleNotFoundError: No module named `addict`
- Solution : `pip install addict`

#### ModuleNotFoundError: No module named `triton`
- Install `triton` package for Windows system
  - Step 1 : Download the Windows `triton` package at the [**HuggingFace**](https://hf-mirror.com/madbuda/triton-windows-builds)
  - Step 2 : Install the `triton` package
    ```commandline
    pip install triton-3.0.0-cp312-cp312-win_amd64.whl
    ```

## Script
### run_predict_depth.py
