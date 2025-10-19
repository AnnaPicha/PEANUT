<!--
  README skeleton for PEANUT project
  Replace placeholders (UPPERCASE) and remove comments when you fill them.
-->

<p align="center">
  <img src="https://github.com/user-attachments/assets/df36443e-af81-4ff5-acbb-35521bc64a5f" alt="PEANUT Logo" width="96" height="96" style="vertical-align:middle; margin-right:10px;">
  <span style="font-size:28px; font-weight:600; vertical-align:middle;">PEANUT</span><br>
  <em>Predicting potential Energies with Artificial NeUral neTworks — a computationally efficient approach for modeling potential energy surfaces.</em>
</p>

---

## Table of Contents
- [Project description](#Project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Model / Architecture](#model--architecture)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Project description
<!-- Brief project summary: what it does, why it exists, what problem it solves. -->
In molecular dynamics (MD) simulations, we can replace classical empirical force fields (FF) with neural network potentials to predict potential energy surfaces. Such networks are usually called neural network potentials (NNP) or machine learning potentials (MLP). A combination of potential energy prediction and a so-called `MD engine’ allows for running simulations of chemical systems such as e.g. a small solute in water, which in turn is a very useful tool to replace e.g. costly lab experiments. This method is an already well-known and established tool in computational chemistry.

Given that molecules can be modeled as graphs where atoms are considered as nodes and edges are considered between interacting atoms, the most recent and successful NNPs are designed as graph neural networks (e.g. DimeNet [1], MACE [2,3]). These graph neural networks learn embeddings of atom types and use graph convolutions or, more general, message passing, to model atom interactions based on interatomic distances. For example the SchNet architecture [4] uses a convolutional neural network architecture for modelling interactions. Other approaches use atom-centered symmetry functions based on distances and angles as feature (ANI [5]). This approach is of course less computational expensive, however these architectures do not use learned features.

Generally, such NNPs are trained on single point energies. Thus, their use in MD simulations is definitely an application outside of their training domain, making the task even more difficult. Also, given that MD simulations are computationally highly expensive, the question of how complex a NNP’s architecture can and should be is crucial. Also, given that the use of NNPs in computational chemistry is still quite new, many methods that are already existing in classical MD simulations (using empirical FFs), need to be re-developed for the use of NNPs. For the development of such methods, the overall accuracy is not always the key point. Often, a functional yet not fully accurate NNP would be sufficient to test new methods.

---

## Features
<!-- Bullet-list of main capabilities -->
- Predicts potential energy surfaces from input XYZ / features
- Fast inference, low compute cost
- Uncertainty estimation (optional)  
- Exportable model (ONNX / TorchScript) (optional)

---

## Installation
<!-- Minimal instructions to get started locally. -->
### Requirements
- Python >= 3.8
- List of main libraries (e.g., torch, numpy, scikit-learn)

### Quick start (example)
```bash
# clone
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO

# create venv (optional)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# install
pip install -r requirements.txt
```
---

## References
[1] https://arxiv.org/abs/2003.03123  
[2] doi.org/10.48550/arXiv.2206.07697  
[3] doi.org/10.1021/jacs.4c07099  
[4] https://arxiv.org/abs/1706.08566  
[5] doi.org/10.1039/C6SC05720A  