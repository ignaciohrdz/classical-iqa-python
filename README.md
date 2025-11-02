# Classical IQA methods

This repository is a collection of classical IQA methods, long forgotten in favour of deep learning-based IQA methods. I am doing this because lots of papers cite classical methods when testing new IQA methods and many of them are not very well documented (GM-LOG, SSEQ, CORNIA, LFA, HOSA...). If all of them were implemented in a single package, it would be easier to try them.

## Implemented methods

- [SSEQ](#spatial-spectral-entropy-based-quality-sseq-index)
- [GM-LOG](#gradient-magnitude-and-laplacian-features-gm-log)
- [ENIQA](#entropy-based-no-reference-iqa-eniqa)
- [LFA](#local-feature-aggregation-lfa)
- [HOSA](#high-orderd-statistics-aggregation-hosa)
- [CORNIA](#codebook-representation-for-no-reference-image-assessment-cornia)
- [SOM](#semantic-obviousness-metric-som)

### Spatial-Spectral Entropy-based Quality (SSEQ) index

This is my implementation of the **SSEQ index**. The full details of SSEQ can be found in the paper: [No-reference image quality assessment based on spatial and spectral entropies (Liu et al.)](https://doi.org/10.1016/j.image.2014.06.006). The original MATLAB implementation is [here](https://github.com/utlive/SSEQ). 

I wasn't able to find a fully implemented Python version of this index, so I decided to use [Aca4peop's code](https://github.com/Aca4peop/SSEQ-Python) as a starting point and then add my own modifications. The main highlight of this version is the vectorized implementation of patch spatial entropy and DCT for spectral entropy (more info [here](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Information_and_Entropy_(Penfield)/03%3A_Compression/3.08%3A_Detail-_2-D_Discrete_Cosine_Transformation/3.8.02%3A_Discrete_Cosine_Transformation))

### Gradient Magnitude and Laplacian features (GM-LOG)

This measure was proposed in [Blind Image Quality Assessment Using Joint Statistics of Gradient Magnitude and Laplacian Features (Xue et al., 2014)](https://ieeexplore.ieee.org/abstract/document/6894197). The authors shared a [MATLAB implementation](http://www4.comp.polyu.edu.hk/~cslzhang/code/GM-LOG-BIQA.zip) that I used as a starting point.

### Entropy-based No-reference IQA (ENIQA)

This one was tougher to implement. The authors shared [their MATLAB implementation](https://github.com/jacob6/ENIQA), but the code was not very well documented and the paper doesn't help either (the explanation of the key fatures doesn't go too deep, and the features are computed in different order than in their MATLAB code!).

In fact, I think their work is flawed because they trained their models on LIVEIQA and tested them on TID2013... And both datasets have some images that are identical. But that's another story and clearly out of the scope of this repository.

### Local Feature Aggregation (LFA)

This measure was proposed in [Local Feature Aggregation for Blind Image Quality Assessment (Xu et al. 2015)](https://ieeexplore.ieee.org/abstract/document/7457832), and it was the precursor of other measures (like HOSA). There are some things to consider:

- Using **16-bit precision** whenever possibe: The construction of the visual codebook is memory-hungry, and probably not intended to be done with a laptop. Each local feature corresponds to a BxB patch, which results in (HxW)/(BxB) patches, and that can take a lot of RAM if you are using a large image dataset. For example, if we resized the images from KonIQ-10k to make them 512x382 and used 7x7 patches, each image would produce 3992 local features, which would result in more than 39M features for the whole dataset! Just imagine if we used the original image size...
- **Image resizing**: Related to my first point, it's not clear whether the images undergo any resizing or if LFA was designed to work for all image sizes. To make it comparable to other IQA measures in this repository, I'm resizing the images to 512px prior to feature extraction.
- **Mini-batch K-means**: If we use larger IQA datasets, K-means will be very slow, so I decided to try this variation ([available in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)) to see if it can speed up the learning phase.


### High Order Statistics Aggregation (HOSA)

I implemented HOSA according to [Blind Image Quality Assessment Based on High
Order Statistics Aggregation (Xu et al.)](https://ieeexplore.ieee.org/document/7501619). 
As it follows a similar approach (generating a codebook and computing some metrics with each cluster's assignments), I've also implemented it with the same tricks I used for LFA.

### Codebook Representation for No-Reference Image Assessment (CORNIA)

CORNIA is a very famous no-reference IQA measure that also makes use of visual codebooks. It was presented in [Unsupervised Feature Learning Framework for No-reference Image Quality Assessment (Ye et al., 2012)](https://ieeexplore.ieee.org/document/6247789) and, after reading the other papers more carefully, I discovered that CORNIA was the starting point for many codebook-based methods (SOM, LFA, HOSA...).

### Semantic Obviousness Metric (SOM)

This one was presented in [SOM: Semantic Obviousness Metric for Image Quality Assessment (Zhang et al., 2015)](https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_SOM_Semantic_Obviousness_2015_CVPR_paper.pdf). This measure uses the BING model for saliency detection. You can find the BING model files in [this repo](https://github.com/methylDragon/opencv-python-reference/tree/master/Resources/Models/bing_objectness_model).

Unfortunately, the paper did not provide enough details for the implementation, such as the computing requirements or any link to their code. I had to make some assumptions and tweak some parameters to make it less memory-hungry, and also use 16-bit precision and mini-batch K-means.

Moreover, the OpenCV implementation of the BING object-like detector doesn't seem to work as expected either, as it is much slower than [what the authors of BING reported (300fps)](https://mmcheng.net/mftp/Papers/ObjectnessBING.pdf). After a quick search, this appears to have happenned to other people in the past, and the OpenCV documentation for these saliency models is useless and almost non-existent. The objectness scores don't seem to work either, and [it has been reported to the opencv-contrib team](https://github.com/opencv/opencv_contrib/issues/404) for a long, long time.

## How to train an IQA model

For all these models I'm following the same approach: splitting every dataset into a training and a test set. I use the training sets with K-fold cross-validation to get the best parameters for each regression model. As of today, it's possible to fit an SVR or an MLP.

I'm using the following datasets:

- [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html)
- [Kadid-10k](https://database.mmsp-kn.de/kadid-10k-database.html)
- [NITSIQA](https://drive.google.com/drive/folders/0B_bnn8Xh3PMmT1VxSlVRWDNCTk0?resourcekey=0-9JzjQxVUNJXIodLwkiZ-Lg&usp=sharing)
- [LIVE-IQA](https://qualinet.github.io/databases/image/live_image_quality_assessment_database/)
- [TID2013](https://qualinet.github.io/databases/image/tampere_image_database_tid2013/)
- [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/)
- [CID:IQ](https://folk.ntnu.no/mariupe/CIDIQ.zip)

## Results

You can find some results [here](./results.json).