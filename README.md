# sexy_yeast
A repository containing code for a simulation of evolution in a sexually reproducing yeast culture.


### **Overview of the Program**

This program simulates **adaptive evolution** of a population modeled using a **Sherrington-Kirkpatrick (SK) spin-glass model** from physics. It introduces **genome evolution** as a series of **mutation events**, where each mutation can either increase or decrease the fitness of the genome.
x
The program uses concepts like:
- **Spin systems (σ)** to represent the state of each genome site.
- **Fitness landscape**, defined by external fields (`h`) and coupling interactions (`J`).
- **Fitness optimization** via a mutation process where each mutation either improves or maintains fitness until the optimal state is reached.
  
The **goal** is to simulate the optimization of a genome (represented by spins ±1) through beneficial mutations and track how fitness evolves until a fitness peak is reached.

---

### **Key Concepts**
1. **Spin System (σ)**
   - Each gene (or spin) is represented as `+1` or `-1`.
   - The configuration of all spins is the state of the genome.

2. **Fitness Calculation**
   - Fitness is a function of external fields (`h`) and interactions between spins (`J`).
   - Mutations flip individual spins, potentially improving fitness.
   - Fitness is maximized through a series of beneficial mutations (Sequential Selection with Weak Mutation, or **SSWM** regime).

3. **Optimization**
   - The main function, `relax_sk`, implements the **optimization loop** that continuously selects beneficial mutations until no further improvements are possible.

---

### **Program Breakdown**

1. **Initialization**
   - The size of the genome (`N = 1000`) and parameters for sparsity (`ρ = 0.25`) and epistatic strength (`β = 0.5`) are defined.
   - These parameters control how strongly genome sites interact and how many of the interaction terms (`J`) are non-zero.

```python
sig_0 = cmn.init_sigma(N)
h = cmn.init_h(N, beta)
J = cmn.init_J(N, beta, rho)
```

- `init_sigma(N)`: Randomly initializes the genome configuration (`+1` or `-1`) for all sites.
- `init_h(N, beta)`: Generates external fields that affect each site’s fitness.
- `init_J(N, beta, rho)`: Initializes a sparse symmetric matrix representing interactions between genome sites.

---

2. **Compute Initial Fitness**
```python
F_off = cmn.calc_F_off(sig_0, h, J)
init_fit = cmn.compute_fit_slow(sig_0, h, J, F_off)
```

- `calc_F_off`: Calculates a fitness offset to normalize fitness values.
- `compute_fit_slow`: Calculates the fitness of a given configuration using external fields (`h`) and interaction matrix (`J`).

---

3. **Optimization Loop**
```python
flip_seq = cmn.relax_sk(sig_0, h, J)
```

- `relax_sk` is the main optimization function:
  - It continuously selects beneficial mutations using **SSWM** until no more beneficial mutations are possible.
  - It returns a sequence of indices representing the sites that were mutated.

---

4. **Reconstruct Final Genome State**
```python
sig_final = cmn.compute_sigma_from_hist(sig_0, flip_seq, num_of_muts)
final_fit = cmn.compute_fit_slow(sig_final, h, J, F_off)
```

- `compute_sigma_from_hist` reconstructs the genome configuration at any point in the mutation history.
- `compute_fit_slow` calculates the fitness of the final configuration.

---

5. **Distribution of Fitness Effects (DFE)**
```python
dfe = cmn.calc_dfe(sig_final, h, J)
```

- `calc_dfe` calculates the **distribution of fitness effects** for all possible single-site mutations in the final genome state.
- The **DFE** shows whether most mutations are beneficial, neutral, or deleterious.

---

### **Logging Results**
The program logs all the important steps:
- Initial and final genome states (`sig_0`, `sig_final`)
- Parameters (`N`, `rho`, `beta`)
- Fitness values at each step
- Mutation sequence (`flip_seq`)
- The distribution of fitness effects (`dfe`)

Example log output:
```
N: 1000
rho: 0.25
beta: 0.5
init_fit: 1.0
num_of_muts: 300
final_fit: 1.45
dfe: [-0.2, 0.1, -0.5, 0.3, ...]
```

---

### **Functions Explained**

1. **Fitness Calculation**
   - `calc_basic_lfs`: Computes the local fitness fields for each site.
   - `calc_energies`: Calculates how much each site contributes to the overall fitness.
   - `compute_fit_slow`: Computes the total fitness of a configuration by summing contributions from `h` and `J`.

2. **Mutation Process**
   - `calc_bdfe`: Calculates the beneficial distribution of fitness effects.
   - `sswm_flip`: Chooses a beneficial mutation based on probabilities derived from the fitness effect of each possible mutation.

3. **Optimization**
   - `relax_sk`: Implements the **SSWM regime**, continuously flipping spins until no beneficial mutations remain.
   - `compute_sigma_from_hist`: Reconstructs the genome state from the mutation history.

---

### **Flow of the Program**
1. **Initialization**
   - Randomly initialize genome (`σ`), external fields (`h`), and interactions (`J`).
   
2. **Compute Initial Fitness**
   - Calculate fitness for the initial genome state.

3. **Optimize Genome**
   - Run the optimization loop (`relax_sk`), flipping beneficial spins until fitness is maximized.

4. **Analyze Results**
   - Compute the final fitness and the distribution of fitness effects (`dfe`).

---

### **Main Goals of the Program**
1. Simulate how a genome evolves by accumulating beneficial mutations.
2. Track how fitness changes during this process.
3. Analyze the **distribution of fitness effects (DFE)** to understand evolutionary dynamics.

---

### **Real-World Applications**
This model can be used in **evolutionary biology** to:
- Study adaptive evolution in complex fitness landscapes.
- Model epistasis (interactions between genes).
- Investigate the role of sparse gene interactions in evolutionary dynamics.

If you'd like, I can run a simulation based on these functions or visualize the fitness landscape for you!