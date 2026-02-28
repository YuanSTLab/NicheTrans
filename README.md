# NicheTrans: Spatial-aware Cross-omics Translation  (Not finished yet)
This is the *official* Pytorch implementation of "NicheTrans: Spatial-aware Cross-omics Translation". 

## Pipeline
![framework](overall.png)

## Preparation
### Installation
```bash
pip install -r requirements.txt
(Here, we list some packages crucial for model reproduction.）
```

### Prepare Datasets
(1) Spatial Multimodal Analysis (SMA) dataset of Parkinson's Disease from ['Spatial multimodal analysis of transcriptomes and metabolomes in tissues'](https://www.nature.com/articles/s41587-023-01937-y). 

(2) STARmap PLUS dataset of Alizimizer's Disease from ['Integrative in situ mapping of single-cell transcriptional states and tissue histopathology in a mouse model of Alzheimer’s disease'](https://www.nature.com/articles/s41593-022-01251-x).

(3) Human breast cancer dataset from ['High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis'](https://www.nature.com/articles/s41467-023-43458-x). 

(4) Human lymph node dataset from ['Deciphering spatial domains from spatial multi-omics with SpatialGlue'](https://www.nature.com/articles/s41592-024-02316-4).

(5) MISAR-seq dataset from ['Simultaneous profiling of spatial gene expression and chromatin accessibility during mouse brain development'](https://www.nature.com/articles/s41592-023-01884-1).

The detailed processing pipelines for the SMA and STARmap PLUS datasets are located in the '1_Data_preparation' folder.
Here, we released the processed spatial multi-omics data at ['GoogleDrive'](https://drive.google.com/drive/folders/1YKBM-N4bP6WJyQ07EmRoZI5lQl0EKFF6?usp=drive_link). Please feel free to reach out to us if the link has expired. 

## Model training & testing
For each spatial multi-omics dataset, we have prepared the Jupyter files for model training and testing separately. __Before model training & testing, please change the direction of the datasets in the 'args' folder.__

The detailed correspondence between subfigures in the manuscript and Jupyter files is shown in:
```bash
codes_figures.txt 
```

Taking the SMA dataset as an example, the training process of NicheTrans is shown in:
```bash
Tutorial 3.1 Train NicheTrans on SMA data.ipynb
```
We provide the jupyter notebook for quantitative evaluation using pcc, spcc, and qualitative visualization of the translated results. 
```bash
Tutorial 3.2 Visualize results.ipynb
```
## Model attribution analysis
Apart from the spatial cross-omics translation, we also provided guidelines for attribution analysis in 'Tutorial 3.3', 'Tutorial 6.5', and 'Tutorial 6.6'.

'Tutorial 3.3' and 'Tutorial 6.5' are for niche-level attribution analysis across genes. 

'Tutorial 6.6' can reveal intercellular cross-omics interaction in the spatial context. 

## Contact
If you have any questions, please don't hesitate to contact us. E-mail: [zkwang00@gmail.com](mailto:zkwang00@gmail.com); [zhiyuan@fudan.edu.cn](mailto:zhiyuan@fudan.edu.cn).
