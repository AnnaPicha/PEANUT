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
  - [Planned workplan](#planned-workplan)
- [Model / Architecture](#model--architecture)
  - [Building blocks](#building-blocks)
  - [Explanation](#explanation)
- [Dataset](#dataset)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Quick start](#quick-start)
- [References](#references)
- [License](#license)

---

## Project description
<!-- Brief project summary: what it does, why it exists, what problem it solves. -->
I believe my project description matches the project type ''beat the stars'' the most. 

---

In molecular dynamics (MD) simulations, we can replace classical empirical force fields (FF) with neural network potentials to predict potential energy surfaces. Such networks are usually called neural network potentials (NNP) or machine learning potentials (MLP). A combination of potential energy prediction and a so-called `MD engine’ allows for running simulations of chemical systems such as e.g. a small solute in water, which in turn is a very useful tool to replace e.g. costly lab experiments. This method is an already well-known and established tool in computational chemistry.

Given that molecules can be modeled as graphs where atoms are considered as nodes and edges are considered between interacting atoms, the most recent and successful NNPs are designed as graph neural networks (e.g. [DimeNet](https://arxiv.org/abs/2003.03123), [MACE](https://arxiv.org/abs/2206.07697)). These graph neural networks learn embeddings of atom types and use graph convolutions or, more general, message passing, to model atom interactions based on interatomic distances. For example the [SchNet](https://arxiv.org/abs/1706.08566) architecture uses a convolutional neural network architecture for modelling interactions. Other approaches use atom-centered symmetry functions based on distances and angles as feature ([ANI](doi.org/10.1039/C6SC05720A)). This approach is of course less computational expensive, however these architectures do not use learned features.

Generally, such NNPs are trained on single point energies. Thus, their use in MD simulations is definitely an application outside of their training domain, making the task even more difficult. Also, given that MD simulations are computationally highly expensive, the question of how complex a NNP’s architecture can and should be is crucial. Also, given that the use of NNPs in computational chemistry is still quite new, many methods that are already existing in classical MD simulations (using empirical FFs), need to be re-developed for the use of NNPs. For the development of such methods, the overall accuracy is not always the key point. Often, a functional yet not fully accurate NNP would be sufficient to test new methods.

Therefore, my goal is to find a possible NNP architecture that imposes an acceptable trade-off between accuracy and computational effort by using and fine-tuning well-established and tested methods to predict potential energy surfaces. 

Since some ideas and key features need to be used in any NNP architecture (e.g. the use of a neighborlist.) If the use of a complex tool such as the construction of a neighbor list will be required in my neural network architecture, I will try to use existing implementations (for pytorch-based models, this could be for example the neighbor list implementation of [NNPOps](https://github.com/openmm/NNPOps)).


### Planned workplan

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

### Building blocks
#### Components
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

### Model evaluation

Even though the ultimative goal would be running MD simulations with this model, this is quite challenging. Apart from a fully trained model, a use in MD simulations would also require a full integration into an existing MD engine. And of course also several MD simulations, which take a lot of time to run. Therefore, for this lecture, I plan to test my model on single point conformations predicting single point energies only. If the model looks promising, I might reconsider and try to integrate it into an MD engine. However, this would clearly be to much for the goal of this lecture. So this parts remains optional, depending of this project's outcome.

---

## Dataset

I plan to use already existing datasets for the training of my neural network. The main effort will go into the cunstruction / design, development and implementation of the neural network architecture and the corresponding training. I have not decided yet which dataset I will use. However, there are multiple datasets available that fit my proposed goal (e.g. the [ANI-2x](https://zenodo.org/records/10108942) or the [SPICE](https://www.nature.com/articles/s41597-022-01882-6) dataset). Apart from those two, that have already been used for the training of e.g. the [ANI-2x](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00121) or the [MACE-OFF](https://doi.org/10.1021/jacs.4c07099) model, there are also some other databases that provide suitibable datasets such as [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

---

## Installation
<!-- Minimal instructions to get started locally. -->
### Requirements
This is just a place-holder. There is nothing to install yet.  
This neural network architecture will be built using pytorch.
- Python >= 3.8
- pytorch, ...

### Quick start
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

Model architectures:  

[1] https://arxiv.org/abs/2003.03123  
[2] doi.org/10.48550/arXiv.2206.07697  
[3] doi.org/10.1021/jacs.4c07099  
[4] https://arxiv.org/abs/1706.08566  
[5] doi.org/10.1039/C6SC05720A  

Datasets:  

---

## License
This project is licensed under the [MIT License](LICENSE).