# Resources

The resource-finder stage produced nothing here, so all resources were gathered in this session.

## Papers (`papers/`)
| File | Paper |
|---|---|
| 1803.00195.pdf | Zhu et al., The Anisotropic Noise in SGD |
| 2002.03495.pdf | Xie et al., A Diffusion Theory for Deep Learning Dynamics |
| 2105.09557.pdf | Mori et al., Power-law Escape Rate of SGD |
| 2111.04004.pdf | Ibayashi & Imaizumi, Exponential Escape Efficiency of SGD |
| 2006.15815.pdf | Xie et al., Adaptive Inertia |
| 2010.05627.pdf | Zhou et al., Towards Theoretically Understanding Why SGD Generalizes Better Than Adam |
| 2105.12221.pdf | Şimşek et al., Geometry of the Loss Landscape in Overparameterized NNs |

## Datasets
- **UCI optdigits** (`sklearn.datasets.load_digits`): 1,797 8×8 images, 10 classes, bundled with scikit-learn.
- **Fashion-MNIST** (`datasets/raw_fmnist/`): `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz` from
  `http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/`. We use a 2,000-example subset, PCA-reduced.
  See `datasets/README.md`.

## Code
No baseline repositories were available or needed: `src/` is a from-scratch batched replica simulator.

## API/LLM note
This study runs real SGD/Adam on real networks; no LLM is involved and no API keys were needed or present.
