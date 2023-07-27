# A full-resolution training framework for Sentinel-2 image fusion

[A full-resolution training framework for Sentinel-2 image fusion](https://ieeexplore.ieee.org/document/9553199) ([ArXiv](https://soon)) is 
a deep learning method for Pansharpening based on unsupervised and full-resolution framework training.

## Cite FR-FUSE

If you use FR-FUSE in your research, please use the following BibTeX entry.

```
@inproceedings{Ciotola2021,
  author={Ciotola, Matteo and Ragosta, Mario and Poggi, Giovanni and Scarpa, Giuseppe},
  booktitle={2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS}, 
  title={A Full-Resolution Training Framework for Sentinel-2 Image Fusion}, 
  year={2021},
  volume={},
  number={},
  pages={1260-1263},
  doi={10.1109/IGARSS47720.2021.9553199}
}
```
 
## License
Copyright (c) 2021 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/FR-FUSE/LICENSE.txt)
(included in this package) 

## Prerequisites
All the functions and scripts were tested on Windows and Ubuntu O.S., with these constraints:

- Python 3.10.10 
- PyTorch 2.0.0
-  Cuda 11.7 or 11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

- Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads) 
- Create a folder in which save the algorithm
- Download the algorithm and unzip it into the folder or from CLI:

```
git clone https://github.com/matciotola/FR-FUSE
```

- Create the virtual environment with the fr_fuse_environment.yml`

```
conda env create -n fr_fuse_env -f fr_fuse_environment.yml
```

- Activate the Conda Environment

```
conda activate fr_fuse_env
```

- Test it 

```
python main.py -b10 example/10/New_York.tiff -b20 example/20/New_York.tiff -o ./Output_Example
```
