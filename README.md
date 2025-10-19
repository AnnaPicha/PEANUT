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
- [Model / Architecture](#model--architecture)
- [Building blocks](#building-blocks)
- [Dataset](#dataset)
- [Installation](#installation)
- [References](#references)
- [License](#license)

---

## Project description
<!-- Brief project summary: what it does, why it exists, what problem it solves. -->
In molecular dynamics (MD) simulations, we can replace classical empirical force fields (FF) with neural network potentials to predict potential energy surfaces. Such networks are usually called neural network potentials (NNP) or machine learning potentials (MLP). A combination of potential energy prediction and a so-called `MD engine’ allows for running simulations of chemical systems such as e.g. a small solute in water, which in turn is a very useful tool to replace e.g. costly lab experiments. This method is an already well-known and established tool in computational chemistry.

Given that molecules can be modeled as graphs where atoms are considered as nodes and edges are considered between interacting atoms, the most recent and successful NNPs are designed as graph neural networks (e.g. DimeNet [DimeNet](https://arxiv.org/abs/2003.03123  ), MACE [2,3]). These graph neural networks learn embeddings of atom types and use graph convolutions or, more general, message passing, to model atom interactions based on interatomic distances. For example the SchNet architecture [4] uses a convolutional neural network architecture for modelling interactions. Other approaches use atom-centered symmetry functions based on distances and angles as feature (ANI [5]). This approach is of course less computational expensive, however these architectures do not use learned features.

Generally, such NNPs are trained on single point energies. Thus, their use in MD simulations is definitely an application outside of their training domain, making the task even more difficult. Also, given that MD simulations are computationally highly expensive, the question of how complex a NNP’s architecture can and should be is crucial. Also, given that the use of NNPs in computational chemistry is still quite new, many methods that are already existing in classical MD simulations (using empirical FFs), need to be re-developed for the use of NNPs. For the development of such methods, the overall accuracy is not always the key point. Often, a functional yet not fully accurate NNP would be sufficient to test new methods.

---

## Model / Architecture
<!-- Bullet-list of main capabilities -->
Conceptual workflow:  
For each atom i:  
    1. Get neighbors in short and medium cutoff ranges  
    2. Compute radial features (learned) and angular features (fixed) for each edge  
    3. Compute attention weights for each neighbor  
    4. Aggregate messages per scale  
    5. Update node embedding h_i  
After N message-passing layers:  
    6. Sum over all nodes to predict molecular energy  

---

## Building blocks
### Components
| **Component**              | **What it does**                                                                                       | **Why it’s important**                               |
|-----------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| **Node features**           | Atom type embeddings, maybe charges, hybridization                                                    | Basis for all calculations                            |
| **Edge features**           | Interatomic distances, optionally radial basis expansion                                              | Encodes pairwise geometry                             |
| **Triplet / angle features**| Angle between bonds for atom triplets                                                                 | Captures directional dependencies                     |
| **Message passing / convolution** | Aggregates neighbor info, possibly with learned weights depending on distance/angle             | Where the network learns chemical interactions        |
| **Update function**         | Updates node features                                                                                 | Allows information to propagate                       |
| **Readout / pooling**       | Converts node embeddings to molecular energy                                                          | Can be sum, mean, or learned aggregation              |

### Explanation

| **Component** | **Explanation** |
|----------------|-----------------|
| **Radial basis** | *RadialBasis* is learnable, taking distances `r_ij` and mapping them to a higher-dimensional embedding. |
| **Angular basis** | *FixedAngularBasis* uses a cosine expansion of angles.<br>No dihedrals needed → cheap to compute. |
| **Edge MLP** | Concatenates sender node, receiver node, and radial + angular features.<br>Outputs a learned message embedding for each edge. |
| **Attention** | Simple sigmoid attention on edges.<br>Could be replaced by softmax per node if desired. |
| **Node update** | Sums messages from neighbors.<br>Passes the result through a small MLP for the new node embedding. |
| **Multi-scale** | Can be implemented by calling this layer separately on different neighbor lists, then summing messages before the node MLP. |


## Installation
<!-- Minimal instructions to get started locally. -->
### Requirements
This is just a place-holder. There is nothing to install yet.
- Python >= 3.8
- pytorch, ...

### Quick start (example)
This is just a place-holder. There is nothing to install yet.

```bash
# clone
git clone git@github.com:AnnaPicha/PEANUT.git
cd PEANUT

# create venv (optional)
conda create -n peanut
conda acticte peanut 

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

---

## License
This project is licensed under the [MIT License](LICENSE).