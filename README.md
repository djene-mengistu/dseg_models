# Deep learning-based automated steel surface defect segmentation: A comparative experimental study
The use of machine vision and deep learning for intelligent industrial inspection has become increasingly important in automating the production processes. Despite the fact that machine vision approaches are used for industrial inspection, deep learning-based defect segmentation has not been widely studied. While state-of-the-art segmentation methods are often tuned for a specific purpose, extending them to unknown sets or other datasets, such as defect segmentation datasets, require further analysis. In addition, recent contributions and improvements in image segmentation methods have not been extensively investigated for defect segmentation. To address these problems, we conducted a comparative experimental study on several recent stateof-the-art deep learning-based segmentation methods for steel surface defect segmentation and evaluated them on the basis of segmentation performance, processing time, and computational complexity using two public datasets, NEU-Seg and Severstal Steel Defect Detection (SSDD). In addition we proposed and trained a hybrid transformerbased encoder with CNN-based decoder head and achieved state-of-theart results, a Dice score of 95.22% (NEU-Seg) and 95.55% (SSDD).

The illustration of the general flow of deep learning-based defect segmentation is presented as follows:
<p align="center">
<img src="/figures/general.jpg" width="70%" height="30%">
</p>

**Fig. 1:** Typical representation of deep learning-based defect segmetnation approaches.

The overall framework of the proposed hybrid network of transformer-based encoder and CNN-based decoder is presented as follows:
<p align="center">
<img src="/figures/hybrid-model.jpg" width="70%" height="40%">
</p>

**Fig. 2:** The framework of the proposed hybrid network. We proposed combining the transformer-based encoder with a CNN-based segmentation head adapted from UperNet. The Pyramid Pooling Module
(PPM) extracts multi-scale features, and the hierarchical features are then fused together for richer representation. The concatenated features are then passed through the up-sampling module for final segmentation output.

# Full paper source:
You can read the details about the methods, implementation, and results from the official website at ([https://link.springer.com/article/10.1007/s11042-023-15307-y])

**Please cite ourwork as follows:**
```
@article{sime2024deep,
  title={Deep learning-based automated steel surface defect segmentation: a comparative experimental study},
  author={Sime, Dejene M and Wang, Guotai and Zeng, Zhi and Peng, Bei},
  journal={Multimedia Tools and Applications},
  volume={83},
  number={1},
  pages={2995--3018},
  year={2024},
  publisher={Springer}
}
```
## Python >= 3.6
PyTorch >= 1.1.0
PyYAML, tqdm, tensorboardX
## Data Preparation
Download datasets. There are 2 datasets to download:
* NEU-SEG dataset
* SSDD (Severstal Steel Defect Dataset)

Put downloaded data into the following directory structure:
* data/
    * NEU-Seg/ ... # raw data of NEU-Seg
    * SSDD/ ...# raw data of SSDD 
## Data loading and preparation 

## Selected results
Selected results of the proposed method and other methods in the comparative study are presented as follows.\
The overall framework of the proposed hybrid network of transformer-based encoder and CNN-based decoder is presented as follows:

<p align="center">
<img src="/figures/results_neu.jpg" width="60%" height="40%">
</p>

**Fig. 3:** Performance summary of CNN-based methods on the NEU-Seg dataset.

<p align="center">
<img src="/figures/results_neu_hybrid.jpg" width="60%" height="40%">
</p>

**Fig. 4:** Performance summary of transformer-based and proposed hybrid methods on the NEU-Seg dataset.

<p align="center">
<img src="/figures/dice vs fps backbone.jpg" width="70%" height="50%">
</p>

**Fig. 5:** Mean Dice score vs. FPS for selected models on the NEU-Seg using various backbone architectures.

## Visualization
The visualization of the segmetnation results for selected methods is presented as follows.\
<p align="center">
<img src="/figures/viz_ssdd.jpg" width="80%" height="30%">
</p>

**Fig. 6:** Visualization of the predicted segmentation maps with selected models on the SSDD dataset.


## Contact
For any issue please contact me at djene.mengistu@gmail.com
