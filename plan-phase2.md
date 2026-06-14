Think of this as the "Physics Engine" phase. The team will take the purely data-driven architecture from Phase 1 and constrain it using the fundamental laws of thermodynamics and quantum mechanics, drastically reducing the need for massive, expensive datasets.

---

**Project Objective:** Transform the baseline coordinate-driven model from Phase 1 into a Physics-Informed Neural Network (PINN). Force the neural density field to obey core quantum physical laws—specifically electron conservation and the Poisson equation—via custom, heavily penalized loss functions.
**Constraint:** The underlying local relative coordinate architecture established in Phase 1 must remain intact. The engineering focus shifts entirely to the training regime, advanced automatic differentiation (`autograd`), and multi-objective optimization.

## Tech Stack & Prerequisites
*   **Deep Learning Framework:** PyTorch (crucially relying on `torch.autograd` for calculating the Laplacian).
*   **Chemistry Engines:** PySCF (for generating a small set of high-fidelity CCSD(T) benchmarks) and `xtb-python` (for rapidly generating the cheap semi-empirical baseline data).
*   **Dataset:** The same QM9 subset used in Phase 1, but utilizing only a fraction (e.g., 10%) of the high-fidelity labels.
*   **Prerequisite:** A fully functioning, SE(3)-invariant baseline model and dataloader from Phase 1.

---

## Milestone 1: The Normalization Constraint (Weeks 1–5)
*Goal: Implement a loss function that ensures the total predicted electron density perfectly integrates to the exact number of electrons in the molecule ($N_{elec}$).*

*   **Task 1.1: Monte Carlo Integration Engine.** Integrating density over a 3D grid is $O(N^3)$ and will bottleneck training. Instead, build a highly optimized Monte Carlo integration module that samples $M$ random points within the molecule's bounding volume.
*   **Task 1.2: Importance Sampling.** Enhance the Monte Carlo engine to sample heavily near the atomic nuclei (where the variance in density is highest) and sparsely in the tails. This significantly reduces the error of the integral estimation.
*   **Task 1.3: Normalization Loss Term ($L_{norm}$).** Calculate the estimated total electrons $\hat{N}_{elec}$ from the Monte Carlo sum. Define the loss as $L_{norm} = (\hat{N}_{elec} - N_{elec})^2$. Integrate this differentiable term into the PyTorch backward pass.

## Milestone 2: Automatic Differentiation & The Poisson Equation (Weeks 6–12)
*Goal: Penalize the AI if the Laplacian of its predicted electrostatic potential does not match its predicted charge density, forcing the network to learn the underlying differential equations.*

*   **Task 2.1: Dual-Output Head.** Modify the decoder of the Phase 1 network. Instead of only predicting the scalar density $\rho(r)$, add a second head to simultaneously predict the scalar electrostatic potential $V(r)$.
*   **Task 2.2: Autograd Laplacian ($\nabla^2$).** Write a custom PyTorch function that uses `torch.autograd.grad` to compute the second-order spatial derivatives of the predicted potential $V(r)$ with respect to the input coordinates $(x, y, z)$. Sum these to get the Laplacian $\nabla^2 V(r)$.
*   **Task 2.3: Poisson Loss Term ($L_{poisson}$).** Construct the core physics loss function: $L_{poisson} = \text{MSE}(\nabla^2 V(r), -4\pi\rho(r))$.
*   **Task 2.4: VRAM Profiling & Optimization.** Computing double derivatives via autograd creates massive computation graphs. Profile the backward pass memory usage. Optimize batch sizes, utilize gradient checkpointing, or use mixed precision to ensure the training loop fits within standard GPU VRAM.

## Milestone 3: $\Delta$-Learning Hybridization Pipeline (Weeks 13–18)
*Goal: Combine cheap, low-fidelity physics with the neural network's ability to learn the "delta" (the difference) to achieve high-fidelity accuracy without thousands of CCSD(T) calculations.*

*   **Task 3.1: Low-Fidelity Data Generation.** Run GFN2-xTB calculations on the QM9 subset to rapidly generate baseline electron densities ($\rho_{low}$). This takes seconds per molecule.
*   **Task 3.2: Delta-Learning Architecture.** Modify the network to act as a residual predictor. Instead of predicting the absolute density, it predicts the correction: $\Delta\rho(r) = \rho_{high}(r) - \rho_{low}(r)$. The AI only needs to learn the complex electron correlation/dispersion physics that xTB misses.
*   **Task 3.3: Pre-training Regime.** Pre-train the base network entirely on the massive, cheap GFN2-xTB dataset to initialize the weights with sensible chemistry logic before activating the expensive physics losses.

## Milestone 4: PINN Fine-Tuning & Validation (Weeks 19–24)
*Goal: Combine all loss terms and prove that the Physics-Informed model can match the Phase 1 baseline's accuracy while using a fraction of the data.*

*   **Task 4.1: Composite Loss Function.** Combine the standard data-driven MSE loss ($L_{data}$), the normalization loss ($L_{norm}$), and the Poisson loss ($L_{poisson}$) into a single dynamic objective: $L_{total} = \lambda_1 L_{data} + \lambda_2 L_{norm} + \lambda_3 L_{poisson}$.
*   **Task 4.2: Hyperparameter Sweeps (The $\lambda$ Balancing Act).** Systematically tune the weighting parameters ($\lambda_1, \lambda_2, \lambda_3$). If the physics constraints are too heavy, the model might ignore the actual data; if too light, the model reverts to a purely data-driven regime.
*   **Task 4.3: The Data-Efficiency Benchmark.** Train the Phase 1 purely supervised model on 100% of the high-fidelity QM9 data. Train the Phase 2 PI-NDF model on only 10% of the high-fidelity data.
    *   *Success Metric:* The PI-NDF model must match or exceed the accuracy of the Phase 1 model, proving that enforcing the laws of physics successfully replaces the need for massive data.

---

## Definition of Done (DoD) for Phase 2
The Phase is complete and ready to advance to Phase 3 (Infinite-Resolution Validation) when:
1. `torch.autograd` successfully and accurately calculates the spatial Laplacian of the predicted potential without triggering Out-Of-Memory (OOM) errors.
2. The Monte Carlo integration successfully estimates the total number of electrons with <1% error compared to the exact analytical integer.
3. The final composite model (using $\Delta$-learning + PINN loss) achieves a test-set accuracy on QM9 comparable to a purely supervised model, but using **90% less** high-fidelity training data.
4. The trained model respects physical realities (e.g., never predicting an impossible total electron count for a given molecule).
