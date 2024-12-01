# Classical IQA methods

This repository is a collection of classical IQA methods long forgotten in favour of deep learning-based IQA methods. I am doing this because lots of papers cite classical methods when testing new IQA methods and the most famous ones are not very well documented (GM-LOG, SSEQ, CORNIA, HOSA, LFA...). If all of them are implemented in a single package, it should be easier to try them.

I've implemented these methods:

- [SSEQ](#sseq)

## Spatial-Spectral Entropy-based Quality (SSEQ) index {#sseq}

This is my implementation of the **SSEQ index**. I wasn't able to find a fully implemented Python version of this index, so I decided to use [Aca4peop's code](https://github.com/Aca4peop/SSEQ-Python) as a starting point and then add my own modifications.

The full details of SSEQ can be found in the paper: [No-reference image quality assessment based on spatial and spectral entropies (Liu et al.)](https://doi.org/10.1016/j.image.2014.06.006). The original MATLAB implementation is [here](https://github.com/utlive/SSEQ).

### Highlights

Vectorized implementation of:

- Patch spatial entropy
- DCT for spectral entropy (more info [here](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Information_and_Entropy_(Penfield)/03%3A_Compression/3.08%3A_Detail-_2-D_Discrete_Cosine_Transformation/3.8.02%3A_Discrete_Cosine_Transformation))

### Results

Every dataset was split into a training and a test set. I used the training sets with K-fold cross-validation to get the best parameters for each SVR model. The following are the results on each test set:

| Dataset  | LCC    | SROCC   |
|----------|--------|---------|
| csiq     | 0.8493 | 0.7913  |
| kadid10k | 0.6075 | 0.5716  |
| koniq10k | 0.5745 | 0.5573  |
| liveiqa  | 0.8726 | 0.8713  |
| tid2013  | 0.7892 | 0.7204  |