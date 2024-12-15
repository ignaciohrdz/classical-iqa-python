# Classical IQA methods

This repository is a collection of classical IQA methods long forgotten in favour of deep learning-based IQA methods. I am doing this because lots of papers cite classical methods when testing new IQA methods and many of them are not very well documented (GM-LOG, SSEQ, CORNIA, LFA, HOSA...). If all of them are implemented in a single package, it would be easier to try them.

## Implemented methods

- [SSEQ](#spatial-spectral-entropy-based-quality-sseq-index)
- [HOSA](#high-orderd-statistics-aggregation-hosa)
- [LFA](#local-feature-aggregation-lfa)

### Spatial-Spectral Entropy-based Quality (SSEQ) index

This is my implementation of the **SSEQ index**. I wasn't able to find a fully implemented Python version of this index, so I decided to use [Aca4peop's code](https://github.com/Aca4peop/SSEQ-Python) as a starting point and then add my own modifications.

The full details of SSEQ can be found in the paper: [No-reference image quality assessment based on spatial and spectral entropies (Liu et al.)](https://doi.org/10.1016/j.image.2014.06.006). The original MATLAB implementation is [here](https://github.com/utlive/SSEQ). The main highlight of this version is the vectorized implementation of:

- Patch spatial entropy
- DCT for spectral entropy (more info [here](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Information_and_Entropy_(Penfield)/03%3A_Compression/3.08%3A_Detail-_2-D_Discrete_Cosine_Transformation/3.8.02%3A_Discrete_Cosine_Transformation))

### High Orderd Statistics Aggregation (HOSA)

I implemented HOSA according to [Blind Image Quality Assessment Based on High
Order Statistics Aggregation (Xu et al.)](https://ieeexplore.ieee.org/document/7501619). However, there were a couple of points in the paper that were not very clear, so I had to make some decisions:

- The construction of the visual codebook is memory-hungry, and probably not meant to be done with a laptop. Each local feature corresponds to a BxB patch, which results in (HxW)/(BxB) patches, and that can take a lot of RAM if you are using a large image dataset. For example, if we resized the images from KonIQ-10k to make them 512x382, each image would produce 3942 local features, which would result in more than 39M features for the whole dataset! Imagine if we used the original image size...
- Related to my first point, it's not clear whether the images undergo any resizing or if HOSA was designed to work for all image sizes. To make it comparable to other IQA measures I implement, I'm resizing the images to 512px prior to feature extraction.

### Local Feature Aggregation (LFA)

One year before HOSA, the same authors presented LFA in [Local Feature Aggregation for Blind Image Quality
Assessment (Xu et al. 2015)](https://ieeexplore.ieee.org/abstract/document/7457832), which can be considered the precursor of HOSA. As it follows a similar approach (generating a codebook and computing some metrics with each cluster's assignments), I've also implemented it.

## How to train an IQA model

For all these models I'm following the same approach: splitting every dataset into a training and a test set. I use the training sets with K-fold cross-validation to get the best parameters for each SVR model.

I'm using the following datasets:

- [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html)
- [Kadid-10k](https://database.mmsp-kn.de/kadid-10k-database.html)
- [NITSIQA](https://drive.google.com/drive/folders/0B_bnn8Xh3PMmT1VxSlVRWDNCTk0?resourcekey=0-9JzjQxVUNJXIodLwkiZ-Lg&usp=sharing)
- [LIVE-IQA](https://qualinet.github.io/databases/image/live_image_quality_assessment_database/)
- [TID2013](https://qualinet.github.io/databases/image/tampere_image_database_tid2013/)
- [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/)