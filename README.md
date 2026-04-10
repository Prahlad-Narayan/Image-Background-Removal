# Image Background Removal using HPC Techniques

This project demonstrates parallelized image background removal using high-performance computing (HPC) techniques. It evaluates multiple parallelization strategies for preprocessing and training a BEiT-large segmentation model on a large dataset of 22,939 images.

## Features

- **Parallel Preprocessing**: Compare multiprocessing, Dask, and Joblib for image preprocessing
- **Parallel Training**: PyTorch multi-threaded training with Intel MKL optimization
- **Model Evaluation**: Quantitative metrics and qualitative visualizations
- **Comprehensive Analysis**: Performance benchmarking and bottleneck analysis

## Parallelization Methods

1. Multiprocessing (ProcessPoolExecutor)
2. Dask (Task-based parallelism)
3. Joblib (loky and threading backends)
4. PyTorch Multi-threading (Intel MKL)

## Dataset

- 22,939 images (10GB+)
  - 500 4K images from P3M-10k
  - 22,439 128×128 portrait images

## Model

- BEiT-large (441M parameters)
- Input size: 640×640
- Task: Semantic segmentation for background removal

## Key Results

- **Best Preprocessing**: Dask with 3.60× speedup (28 workers)
- **Training Speedup**: 1.48× from 16 to 28 CPUs
- **Model Performance**: 94.9% IoU, 97.4% F1-score

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Dask
- Joblib
- NumPy, Pandas, Matplotlib, Pillow

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook image_background_removal.ipynb`

## Usage

Run the cells in order from 1-16. The notebook includes:

- Configuration and setup
- Serial baseline implementation
- Parallel preprocessing comparisons
- Model training and evaluation
- Results visualization and analysis

## Project Structure

```
.
├── image_background_removal.ipynb    # Main notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
└── results/                          # Output directory (created by notebook)
    ├── models/                       # Trained models
    ├── visualizations/               # Plots and images
    ├── data/                         # Processed data
    └── summaries/                    # CSV and JSON results
```

## System Requirements

- 28+ CPUs recommended
- 32GB+ RAM
- Storage for 10GB+ dataset

## License

This project is open-source. Please cite if used in research.

## Author

Prahlad Narayan