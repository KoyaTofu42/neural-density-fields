
Think of this as an AI engineering "Epic" designed to take the team from a blank slate to a fully functional, SE(3)-invariant neural density field baseline within 6 months.

---

**Project Objective:** Build a grid-free, coordinate-driven neural network that predicts the electron density of a molecule at any arbitrary point in 3D space, strictly using the relative distances to nearby atomic nuclei.
**Constraint:** This phase will *not* include the physics-informed loss (PINNs) yet. It will be trained using standard supervised learning on a small dataset of pre-calculated DFT electron densities to prove the architecture works.

## Tech Stack & Prerequisites
*   **Deep Learning Framework:** PyTorch & PyTorch Geometric (PyG).
*   **3D/Equivariant ML Libraries:** SchNetPack or e3nn (for the baseline Message Passing Network).
*   **Quantum Chemistry Engine:** PySCF or Psi4 (to generate the ground-truth electron density training data).
*   **Dataset:** A subset of QM9 (e.g., 1,000 small organic molecules).

---

## Milestone 1: Data Generation & Pipeline Setup (Weeks 1–4)
*Before writing the neural network, the team must generate the continuous spatial data used to train the baseline.*

*   **Task 1.1: Ground Truth Calculation.** Use PySCF to run cheap DFT calculations (e.g., B3LYP/def2-SVP) on 1,000 QM9 molecules to get their exact wavefunction and electron density.
*   **Task 1.2: Point Sampling Engine.** Build a Python script that takes the DFT output and calculates the exact electron density at **randomly sampled continuous spatial coordinates** around the molecule. 
    *   *Strategy:* Sample heavily near the atomic nuclei (where density is high and complex) and sparsely in the empty space further away.
*   **Task 1.3: Dataloader Creation.** Create a PyTorch dataloader where the input is `[Molecule Graph, Sampled Coordinate (X,Y,Z)]` and the target label is `[True Electron Density at that coordinate]`.

## Milestone 2: Atomic Anchor Embedding (Weeks 5–8)
*The network must understand the chemical environment (the "Anchors") before it can process empty space.*

*   **Task 2.1: MPNN Implementation.** Implement a lightweight Message Passing Neural Network (like SchNet or a basic Graph Attention Network). 
*   **Task 2.2: Node Feature Extraction.** Train the MPNN to read the atoms (nodes) and bonds (edges) of the molecule.
*   **Task 2.3: Latent Vector Output.** Ensure the MPNN outputs a continuous "latent vector" (embedding) for *every single atom* in the molecule. This vector now contains the context of the whole molecule, centered on that specific atom.

## Milestone 3: The Local Query Engine (Weeks 9–14)
*This is the core mathematical breakthrough. We must translate global 3D space into local relative physics.*

*   **Task 3.1: $K$-Nearest Neighbor (KNN) Search.** Write a differentiable function that takes an arbitrary spatial coordinate ($r_i$) and identifies the $k$-nearest atomic anchors (e.g., the 3 closest atoms).
*   **Task 3.2: Relative Coordinate Math.** For the queried point $r_i$, calculate its exact Euclidean distance ($d$) to those $k$ atoms. 
*   **Task 3.3: SE(3)-Invariant Feature Construction.** Instead of passing raw $(X, Y, Z)$ coordinates to the AI (which breaks if the molecule rotates), pass only rotational invariants: the distances to the nearest atoms, and the angles (dot products) between them.
*   **Task 3.4: Gaussian Smearing (Positional Encoding).** Map those scalar distances into a higher-dimensional space using Radial Basis Functions (RBFs) or Gaussian smearing. This helps the neural network see microscopic differences in distance.

## Milestone 4: The Continuous Decoder (Weeks 15–20)
*This is the "NeRF" (Neural Radiance Field) equivalent for chemistry. It combines the chemistry context (Milestone 2) with the spatial context (Milestone 3).*

*   **Task 4.1: Feature Concatenation.** For a queried point $r_i$, combine the MPNN latent vectors of its $k$-nearest atoms with the relative distance features calculated in Milestone 3.
*   **Task 4.2: Distance-Weighted Attention.** Implement an attention mechanism so the point pays more "attention" to the embedding of an atom that is 0.5 Å away, and less to an atom 2.0 Å away.
*   **Task 4.3: MLP Prediction.** Pass this combined data through a 4-layer Multi-Layer Perceptron (MLP). The final output layer must be a single neuron with a Softplus activation function (since electron density must be a positive number, $\rho \ge 0$).

## Milestone 5: SE(3) Validation & Baseline Benchmarking (Weeks 21–24)
*Prove that the model successfully bypassed the $O(N^3)$ Voxel Curse and respects 3D physics.*

*   **Task 5.1: Supervised Training Loop.** Train the whole pipeline end-to-end on the QM9 subset using Mean Squared Error (MSE) against the PySCF ground truth data.
*   **Task 5.2: The Rotation Test.** Take a test molecule. Predict the electron density at 100 points. Now, mathematically rotate the molecule's atomic coordinates by 45 degrees, rotate the query points by 45 degrees, and ask the AI to predict again. 
    *   *Success Metric:* The predictions must be 100.0% identical. This proves native SE(3) invariance.
*   **Task 5.3: Infinite Resolution Render.** Write a visualization script (using Mayavi or PyMOL). Instead of querying random points, query a massively dense grid of points localized *only* around a specific carbon-carbon bond to render a hyper-resolution 3D isosurface of the electron cloud.

---

## Definition of Done (DoD) for Phase 1
The Phase is complete and ready to advance to Phase 2 (Physics-Informed Loss) when:
1. The model can accept an arbitrary $(X,Y,Z)$ coordinate near a molecule and output a density value in under 5 milliseconds.
2. The VRAM usage remains flat regardless of how densely the user queries the space (proving the voxel grid is dead).
3. The model achieves an MSE error of less than $10^{-3}$ electrons/Bohr$^3$ on the QM9 test set.
4. The architecture passes the strict SE(3) rotational invariance test.