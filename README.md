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
  - [Suggested workplan](#suggested-workplan)
- [Model / Architecture](#model--architecture)
  - [Key features](#key-features)
  - [Representation learning](#representation-learning)
  - [Tool box](#tool-box)
  - [Challenges](#challenges)
  - [Model evaluation](#model-evaluation)
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

In molecular dynamics (MD) simulations, it is possible to replace classical empirical force fields (FF) with neural network potentials to predict potential energy surfaces. Such networks are usually called neural network potentials (NNP) or machine learning potentials (MLP) - to avoid confusions, I will use NNP and MLP = multilayer perceptron from now on. A combination of potential energy prediction and a so-called ''MD engine'' allows for running simulations of chemical systems such as e.g. a small solute in water, which in turn is a very useful tool to replace costly lab experiments. This method is an already well-known and established tool in computational chemistry.

Given that molecules can be modeled as graphs where atoms are considered as nodes and edges are considered between interacting atoms, the most recent and successful NNPs are designed as graph neural networks (e.g. [DimeNet](https://arxiv.org/abs/2003.03123), [MACE](https://arxiv.org/abs/2206.07697)). These graph neural networks learn embeddings of atom types and use message passing along edges to model atom interactions based on interatomic distances and 3D arrangements. Other architecture types have also been proposed: [SchNet](https://arxiv.org/abs/1706.08566) uses a continuous-filter convolutional neural network architecture for modelling interactions. Other approaches use atom-centered symmetry functions based on distances and angles as features as fixed representation ([ANI](doi.org/10.1039/C6SC05720A)). This approach is of course less computationally expensive, however these architectures do not use representation learning.

Generally, such NNPs are trained on single point energies. Thus, their use in MD simulations is definitely an application outside of their training domain, making the task even more difficult. Also, since MD simulations are computationally highly expensive, the question of how complex a NNP’s architecture can and should be is crucial. Given that the use of NNPs in computational chemistry is still quite new, many methods that are already existing in classical MD simulations (using empirical FFs), need to be re-developed for the use of NNPs. For the development of such methods, the overall accuracy is not always the key point. Often, a functional yet not fully accurate NNP would be sufficient to test new methods.

Therefore, my goal is to propose a NNP architecture that achieves an acceptable trade-off between accuracy and computational effort by using and fine-tuning well-established and tested methods to predict potential energy surfaces. 

Some ideas and key features need to be used in any NNP architecture, such as a neighborlist or a differentiable cutoff function. Therefore, whenever possible, I will try to use existing implementations (for pytorch-based models, this could be for example the neighbor list implementation of [NNPOps](https://github.com/openmm/NNPOps)).

---

### Suggested workplan


| Phase | Timeframe | Goals | Key Tasks | Milestone |
|-------|-----------|-------|-----------|-----------|
| 1. Planning & Setup | Mid Oct – End Oct | Finalize dataset selection, define initial architecture | - Decide on datasets<br>- Set up Python environment and dependencies (PyTorch, data loaders)<br>- Write basic scripts to load data<br>- Draft initial model architecture outline | Environment ready, dataset downloaded, first model outline completed |
| 2. Prototype Architecture | Early Nov – Mid Nov | Implement basic neural network, ensure data flows through the model | - Implement node/edge features and basic message-passing layers<br>- Implement readout/pooling for energy prediction<br>- Run small-scale tests on dataset subset<br>- Debug tensor shapes, feature dimensions, edge cases | Working prototype of neural network producing energy predictions |
| 3. Feature & Block Refinement | Mid Nov – End Nov | Add full set of building blocks, optimize model structure | - Implement radial basis, angular basis, edge MLP fully<br>- Add multi-scale message passing<br>- Implement attention mechanism and node update functions<br>- Run small-scale training to validate stability | Full model implemented with all planned building blocks |
| 4. Training & Evaluation | Early Dec – Mid Dec | Train on larger datasets, evaluate accuracy, define suitable loss function | - Train on small subsets for debugging<br>- Scale up to full dataset<br>- Evaluate using MAE, RMSE<br>- Adjust hyperparameters (learning rate, batch size, layers) | Model trained and evaluated; preliminary results available |
| 5. Optimization & Final Experiments | Mid Dec – End Dec | Improve model performance, compare with benchmarks | - Test alternative architectures if needed<br>- Perform hyperparameter tuning<br>- Document experiments and results<br>(- Optional: run small MD simulations) | Finalized model with experimental results ready |
| 6. Documentation & Report | Early Jan – Mid Jan | Prepare final report, README, and visualizations | - Write detailed README (architecture, dataset, results, usage)<br>- Create diagrams for model workflow and building blocks<br>- Summarize training results, tables, plots<br>- Write additional project documentation | Documentation complete and ready for submission/sharing |
| 7. Buffer & Final Review | Mid Jan – End Jan | Final polish and bug fixes | - Fix any remaining bugs/issues<br>- Re-run experiments if needed<br>- Final proofreading of documentation | Project completed and fully documented by end of January |


---

## Model / Architecture
<!-- Bullet-list of main capabilities -->


## Key features

The model will use learned features (see Section representation learning) to construct node features per atom. These node features are finally processed through a MLP to predict potential energy contributions per atom. Finally, the sum of all atom-wise energy contributions shall match the energy of the whole input data (e.g. energy of a molecule).  

If a model is applied to the above described setting, we have to ensure that several symmetry requirements are fullfilled to ensure that physical properties are conserved. This includes translational invariance, rotational invariance and permutational invariance. 
- Rotational invariance: use distances for radial features and spherical harmonics for angles (or other invariant angular descriptors)  
- Translational invariance: using relative positions only ensures this  
- Permutational invariance: sum or mean over neighbor messages ensures exchangeability  


```markdown
Conceptual workflow:

For each atom i:
1.1. Get neighbors in the cutoff range
1.2. Compute radial and directional features (learned) for each edge
1.3. Compute attention weights for each neighbor (closer neighbors are chimally more important)
1.4. Aggregate messages per scale
1.5. Update node embedding h_i

After N message-passing layers:
2.1. Sum over all nodes to predict molecular energy
```  
Note: In step 1.1., I will have to use the previously mentioned neighbor list. Step 1.4. accounts for the possibility of using more than one edge MLP for different meassge passing treatment, depending on pairwise ditances (''closer atoms are more important'').  

 Symmetry requirements will be fulfilled as to the following sketched overview:

```markdown
[Atomic positions r_i] 
       |
       v
[Neighbor list r_ij]  <-- translational invariance via relative positions
       |
       v
[Radial features r_ij]  <-- rotational & translational invariance
       |
       v
[Directional features Y_ij]  <-- rotational invariance
       |
       v
[Message passing / attention]  <-- permutational invariance
       |
       v
[Node embeddings h_i] 
       |
       v
[Sum/Pooling] --> Energy (invariant)
       |
       v
[Optional: Gradient] --> Forces (equivariant)

```

Note: The potential energy will be used to compute forces that act on each atom (F = -\Nabla U). In case forces will be used during the training, the energy gradients can also provide an auxiliary supervision signal.

---

## Representation learning
The goal of the model is to predict atom-wise energy contributions to a chemical system (e.g. one single molecule). That is, the model needs to learn a suitable representation for each atom in the system. This will be divided in two key parts: Radial and directional features. Radial features will be based on pairwise distances. Directional features will be constructed using vector-based descriptors (this is the part where we need the neighbor list). For both types of features, we can use some fixed initial descriptors such as symmetry functions, spherical harmonics, Bessel functions etc., that are then passed through a learnable MLP. The node vectors will be updated iteratively (-> representation learning). 

Some succesful architectures, such as [MACE](https://doi.org/10.48550/arXiv.2206.07697) or [NequIP](https://www.nature.com/articles/s41467-022-29939-5), explicitly ensure equivariance by using tensor representations instead of vectors. [ANI](doi.org/10.1039/C6SC05720A) uses triplets and fixed angular features to account for 3D-arrangements. However, my goal is to propose a less accurate but hopefully faster architecture. Therefore, I plan to only use a directional edge representation (no atom triplets and no higher-order tensors). This design retains directional awareness at much lower computational cost compared to real equivariant networks. This will surely reduce the network's potential accuracy, but also hopefully reduce computation times.  


---

### Building blocks for representation learning
| **Component**              | **Explanation**                               |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Node features**           | Learned atom type embeddings `h_i`, basis for all calculations                            |
| **Edge features**           | Messages `m_ij`, are constructed with learned node features |
| **Message passing / graph NN** | Aggregates neighbor information, possibly with learned weights depending on distances/directions (attention layer) -> the network learns chemical interactions        |
| **Update function**         | Updates node features  to allow information to propagate                       |
| **Readout / pooling**       | Converts node embeddings to molecular energy. Can be sum, mean, or learned aggregation              |

### Tool box

| **Component** | **Explanation** |
|----------------|-----------------|
| **Radial basis** `b_{ij}`| *RadialBasis* is learnable, taking distances `r_ij` and mapping them to a higher-dimensional embedding. `b_{ij} = MLP(h_i, h_j, \phi(r_ij))`, `\phi` will be some initial descriptor.|
| **Directional basis** `Y_{ij}`| *DirectionalBasis* e.g. Vector-based Spherical harmonics of rank `l` (`Y_{ij} = Y_l(r_{ij})`|
| **Edge MLP** | Concatenates sender node (initially these are the embedding vectors), receiver node, and radial + angular features.<br>Outputs a learned message embedding for each edge. `m_{ij} = MLP(h_i, h_j, b_{ij}, Y_{ij})`|
| **Attention** `\alpha_{ij}` | Simple attention on messages (sigmoid or softmax).<br>Could be replaced by softmax per node if desired. |
| **Node update** | Sums (weighted) messages (`m^{'}_{ij} = \alpha_{ij}\cdot m_{ij}`) from neighbors.<br>Passes the result through a small MLP for the new node embedding. |
| **Multi-scale** (Optional) | Optional use of 2-3 different edge MLPs to allows different treatments of neighboring atoms based on distance. Can be implemented by calling this layer separately on different neighbor lists, then summing messages before the node MLP. |

#### Sketch for attention layer part
```bash
Node i neighbors: j1, j2, j3

Edge MLP: compute m_ij1, m_ij2, m_ij3  <-- message passing
Attention: compute α_ij1, α_ij2, α_ij3  <-- per message
Weighted messages: α_ij * m_ij
Aggregate: sum/mean → updated node embedding h_i

```

### Challenges

I am not sure if the use of attention layers and / or multi-scale edge MLPs are very costly. Also, I am not sure what the profit is. I do want to try using both of them to give near neighbors a higher relevance to the central atoms. But if this does not help a lot or is too costly, I might remove these parts.



### Model evaluation

Even though the ultimative goal would be running MD simulations with this model, this is quite challenging. Apart from a fully trained model, a use in MD simulations would also require a full integration into an existing MD engine. And of course also several MD simulations, which take a lot of time to run. Therefore, for this lecture, I plan to test my model on single point conformations predicting single point energies only. If the model looks promising, I might reconsider and try to integrate it into an MD engine. However, this would clearly be to much for the goal of this lecture. So this parts remains optional, depending of this project's outcome.

---


## Dataset

I plan to use already existing datasets for the training of my neural network. The main effort will go into the cunstruction / design, development and implementation of the neural network architecture and the corresponding training. I have not decided yet which dataset I will use. However, there are multiple datasets available that fit my proposed goal (e.g. the [ANI-2x](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00121) or the [SPICE](https://www.nature.com/articles/s41597-022-01882-6) dataset). Apart from those two, that have already been used for the training of e.g. the [ANI-2x](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00121) or the [MACE-OFF](https://doi.org/10.1021/jacs.4c07099) model, there are also some other databases that provide suitibable datasets such as [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

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
conda activate peanut 

# install
pip install -r requirements.txt
mamba install -c conda-forge pytorch nnpops
```
---

## References

### Model Architectures

1. **[Directional Message Passing for Molecular Graphs](https://doi.org/10.48550/arXiv.2003.03123)**  
   Johannes Gasteiger, Janek Groß, Stephan Günnemann, 2022. *arXiv:2003.03123*

2. **[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://doi.org/10.48550/arXiv.2206.07697)**  
   Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, Gábor Csányi, 2023. *arXiv:2206.07697*

3. **[MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules](https://doi.org/10.1021/jacs.4c07099)**  
   Dávid Péter Kovács, J. Harry Moore, Nicholas J. Browning, Ilyes Batatia, Joshua T. Horton, Yixuan Pu, Venkat Kapil, William C. Witt, Ioan-Bogdan Magdău, Daniel J. Cole, Gábor Csányi, 2025. *DOI: 10.1021/jacs.4c07099*

4. **[SchNet: A Continuous-Filter Convolutional Neural Network for Modeling Quantum Interactions](https://arxiv.org/abs/1706.08566)**  
   Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller, 2017. *arXiv:1706.08566*

5. **[ANI-1: An Extensible Neural Network Potential with DFT Accuracy at Force Field Computational Cost](https://doi.org/10.1039/C6SC05720A)**  
   J. S. Smith, O. Isayev, A. E. Roitberg, 2017. *DOI: 10.1039/C6SC05720A*

6. **[E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials](https://www.nature.com/articles/s41467-022-29939-5)**  
   Batzner, S., Musaelian, A., Sun, L. et al., 2022. *doi.org/10.1038/s41467-022-29939-5*

---

### Datasets

1. **[ANI-2x](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00121)**  
   Extending the Applicability of the ANI Deep Learning Molecular Potential to Sulfur and Halogens  
   Christian Devereux, Justin S. Smith, Kate K. Huddleston, Kipton Barros, Roman Zubatyuk, Olexandr Isayev, Adrian E. Roitberg, 2020. *DOI: 10.1021/acs.jctc.0c00121*

2. **[SPICE](https://www.nature.com/articles/s41597-022-01882-6)**  
   A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials  
   Eastman, P., Behara, P. K., Dotson, D. L., et al., 2023. *DOI: 10.1038/s41597-022-01882-6*

3. **[PubChem](https://pubchem.ncbi.nlm.nih.gov/)**  
   Open Chemistry Database, National Institutes of Health.



---

## License
This project is licensed under the [MIT License](LICENSE).