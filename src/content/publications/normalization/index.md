---
title: "ADTnorm: robust integration of single-cell protein measurement across CITE-seq datasets"
authors:
  - "Ye Zheng"
  - "Daniel P Caron"
  - "Ju Yeong Kim"
  - "Seong-Hwan Jun"
  - "Yuan Tian"
  - "Florian Mair"
  - "Kenneth D Stuart"
  - "Peter A Sims"
  - "Raphael Gottardo"
journal: "Nature Communications 16, 5852"
date: 2025-07-01
type: paper
doi: "10.1038/s41467-025-61023-6"
pdf: "/publications/normalization/2022.04.29.489989v1.full.pdf"
image: "./featured-image.jpg"
---

Cellular Indexing of Transcriptomes and Epitopes by Sequencing (CITE-seq) enables paired measurement of surface protein and mRNA expression in single cells using antibodies conjugated to oligonucleotide tags. Due to the high copy number of surface protein molecules, sequencing antibody-derived tags (ADTs) allows for robust protein detection, improving cell-type identification. However, variability in antibody staining leads to batch effects in the ADT expression, obscuring biological variation, reducing interpretability, and obstructing cross-study analyses. Here, we present ADTnorm, a normalization and integration method designed explicitly for ADT abundance. Benchmarking against 14 existing scaling and normalization methods, we show that ADTnorm accurately aligns populations with negative- and positive-expression of surface protein markers across 13 public datasets, effectively removing technical variation across batches and improving cell-type separation. ADTnorm enables efficient integration of public CITE-seq datasets, each with unique experimental designs, paving the way for atlas-level analyses. Beyond normalization, ADTnorm includes built-in utilities to aid in automated threshold-gating as well as assessment of antibody staining quality for titration optimization and antibody panel selection. Applying ADTnorm to an antibody titration study, a published COVID-19 CITE-seq dataset, and a human hematopoietic progenitors study allowed for identifying previously undetected phenotype-associated markers, illustrating a broad utility in biological applications.
