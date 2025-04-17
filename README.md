# MNIST Explainability Demo

This repository demonstrates instance‐wise explanations on MNIST digits using two methods:

1. **L2X:** An explainer which measures mutual information between features and the model’s output, helping to 
identify the most important subset of input features (e.g., pixels) that retain the predictive power of the original input.
2. **SHAP**: An explainer based on Shapley values from cooperative game theory, which attributes importance scores
to each input feature by estimating its marginal contribution to the model’s output. This helps identify how much each feature (e.g., pixel) pushes the prediction toward or away from a particular class, offering both local and globally consistent explanations.

----

## 📝 Contents

- [`environment.yml`](#setup-environment) – Complete Conda env spec.  
- [`model.py`](#model-definition) – Defines and trains a logistic‑regression model on MNIST.  
- [`util.py`](#data-utilities) – MNIST dataset loader.  
- [`exec.ipynb`](#notebook-demo) – End‑to‑end demo: trains the model, runs L2X & SHAP explainers, and visualizes results inline.  
- [`mnist_png/explained_imgs/`](#example-images) – Ten 28×28 grayscale PNGs used for out‑of‑sample explanations.  

---

## 🚀 Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your‑user>/mnist-explainability.git
   cd mnist-explainability

2. **Decompress MNIST training data**
    ```bash
   tar -xzf mnist_png.tar.gz

3. **Create a Conda Environment**
    ```bash
    conda env create -f environment.yml
    conda activate mnist-explainability

4. **Run**
    ```bash
   jupyter lab ecex.ipynb
   
## Citations
- Chen, J., Song, L., Wainwright, M. J., & Jordan, M. I. (2018). **Learning to Explain: An Information‐Theoretic Perspective on Model Interpretation.** *Proceedings of the 35th International Conference on Machine Learning* (ICML). arXiv:1802.07814

- Lundberg, S. M., & Lee, S.‑I. (2017). **A Unified Approach to Interpreting Model Predictions.** *Advances in Neural Information Processing Systems*, 30, 4765–4774.  

