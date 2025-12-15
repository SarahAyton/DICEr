---
title: "Readme"
author: "Sarah Ayton"
date: "2025-12-15"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# DICEr: Deep significance clustering

**DICEr** is an R package implementing a deep learning framework for representation learning, clustering, and outcome-aware classification using autoencoders, k-means clustering, and joint optimization.  
It is designed for high-dimensional biomedical data, where discovering latent subgroups with distinct outcome profiles is critical.

The method combines:
- Deep autoencoder-based representation learning
- Iterative clustering with outcome-informed relabeling
- Joint optimization of reconstruction, clustering, classification, and outcome losses
- Statistical control via likelihood-based p-value constraints

DICEr is implemented using **torch for R**, enabling GPU acceleration and scalable training.

---

## Key Features

- Outcome-aware clustering with interpretable subgroups  
- Deep representation learning via LSTM-based autoencoders  
- Joint optimization of reconstruction, classification, and outcome prediction  
- Statistical safeguards using likelihood ratio-based p-value constraints  
- GPU-ready via torch  
- Designed for omics, longitudinal, and real-world clinical data  

---

## Installation

```{r}
# Install dependencies
# install.packages(c("torch", "ggplot2", "cluster", "pROC"))

# Install DICEr (after torch is configured)
# remotes::install_github("YiyeZhangLab/DICEr")
```

Note: torch must be properly installed using torch::install_torch().

## Data Format

DICEr expects data serialized as an `.rds` file containing a list of:

1. `data_x`: feature matrix (samples × features)
2. `data_v`: auxiliary or demographic covariates
3. `data_y`: binary outcome (0/1)

This structure reflects common biomedical use cases such as molecular features combined with demographics and clinical outcomes.

## Quick Start Example (Iris Data)
This example demonstrates workflow setup and API usage only. It is not biologically meaningful.

```{r}
library(DICEr)
library(torch)

data(iris)

x <- as.matrix(iris[, 1:4])
v <- matrix(0, nrow(x), 1)
y <- as.integer(iris$Species == "setosa")

saveRDS(
  list(x, v, y),
  file = "iris_train.rds"
)

file.copy("iris_train.rds", "iris_test.rds")
```

## Define Model Arguments
```{r}
args <- list(
  seed = 123,
  input_path = "./",
  filename_train = "iris_train.rds",
  filename_test  = "iris_test.rds",

  n_input_fea = 4,
  n_hidden_fea = 8,
  lstm_layer = 1,
  lstm_dropout = 0.1,
  K_clusters = 2,
  n_dummy_demov_fea = 1,

  cuda = FALSE,
  lr = 1e-3,

  init_AE_epoch = 10,
  iter = 2,
  epoch_in_iter = 5,

  lambda_AE = 1,
  lambda_classifier = 1,
  lambda_outcome = 1,
  lambda_p_value = 0.1
)
```

## Run DICEr
```{r, eval=FALSE}
main(args)
```

## Outputs include:
* Learned latent representations
* Cluster assignments
* Trained model checkpoints
* Loss curves and diagnostics

Output Structure
```{text}
hn_<hidden>_K_<clusters>/
├── part1_AE_nhidden_*/
│   ├── AE_model_*.pt
│   └── part1_loss_AE.png
└── part2_AE_nhidden_*/
    ├── model_iter.pt
    ├── data_train_iter.rds
    └── data_test_iter.rds
```

## When to Use DICEr
DICEr is particularly suited for:
* Cancer subtyping
* Disease progression modeling
* Patient stratification
* Multi-omics clustering
* Longitudinal clinical data

## Computational Notes
* Training is computationally intensive
* GPU acceleration is strongly recommended for real datasets
* Batch size is fixed at 1 by design to support sequence modeling

## Reproducibility
* Explicit random seed control
* Deterministic clustering relabeling
* Full model checkpoints saved to disk
