# Superscan: Supervised Single-Cell Annotation


<!--more-->

## Authors
Carolyn Shasha, ***Yuan Tian***, Florian Mair, Helen ER Miller, Raphael Gottardo

## Journal
bioRxiv

## Abstract
Automated cell type annotation of single-cell RNA-seq data has the potential to significantly improve and streamline single cell data analysis, facilitating comparisons and meta-analyses. However, many of the current state-of-the-art techniques suffer from limitations, such as reliance on a single reference dataset or marker gene set, or excessive run times for large datasets. Acquiring high-quality labeled data to use as a reference can be challenging. With CITE-seq, surface protein expression of cells can be directly measured in addition to the RNA expression, facilitating cell type annotation. Here, we compiled and annotated a collection of 16 publicly available CITE-seq datasets. This data was then used as training data to develop Superscan, a supervised machine learning-based prediction model. Using our 16 reference datasets, we benchmarked Superscan and showed that it performs better in terms of both accuracy and speed when compared to other state-of-the-art cell annotation methods. Superscan is pre-trained on a collection of primarily PBMC immune datasets; however, additional data and cell types can be easily added to the training data for further improvement. Finally, we used Superscan to reanalyze a previously published dataset, demonstrating its applicability even when the dataset includes cell types that are missing from the training set.

> [Download PDF](2021.05.20.445014v1.full.pdf)
