## Ground states of frustrated magnetic systems
Developing new methods to find and analyse ground states of frustrated magnetic systems using methods of discrete mathematics and neural networks

### Project description

We are developing new methods to find and analyse ground states of frustrated magnetic systems. Technically, it means that we need to find eigenvectors of very large (~2^{50} × 2^{50}) sparse matrices. As the size increases, straightforward application of classical diagonalization algorithms becomes unfeasible. Alternative approaches that are currently used in the field represent the eigenvector as a parametrized function, using either some explicit formula or a neural network, and then rewrite the problem as an optimization one. Unfortunately, this kind of methods demonstrate poor performance for highly-frustrated systems. We are going to use several new ideas to better understand the structure of the ground states and develop new methods to find them. Specifically, we are going to use Boolean Fourier analysis to study sign structures of the ground states and a combination of discrete optimization methods and novel neural network architectures to do better optimization. Thus we need to perform the following types of computation:

1. Find exact diagonalizations to obtain benchmarks and training sets.
2. Apply Boolean Fourier transform (Hadamard transform) to large vectors.
3. Perform training of various neural network architectures in supervised and unsupervised (variational Monte-Carlo) settings.
4. Perform discrete optimization of sign structures

### Project requirements
Compute

We require 1000,000 SBUs according to the following preliminary calculations.

The physical systems we are studying are Heisenberg spin-1/2 AFM Hamiltonians on three types of lattices (square, triangle, kagome) of two sizes (24 spins and 36 spins) with 20 values of a parameter. The most demanding part of our project is training neural networks. To demonstrate that our methods deliver state of the art results, we have to do extensive architecture and hyperparameters search. We will use 1/4 of gcn node (1 GPU).

Thus we need: 6 (lattices) × 20 (parameters) × 20 (architectures) x 3 (learning rates) × 1.4 (core-hour per run) × 7.11 (SBU weight) × (2 settings) × 5 (runs) = 716,688 SBUs on GPU.

We also need to perform CPU-based calculations:

- Exact diagonalization. We performed preliminary tests on our local epyc node with smaller systems  with lattice-symmetries library that demonstrate good scaling (https://arxiv.org/abs/2308.16712). According to them, we need 2.5 hours × 128 cores × 3 (large lattices) × 20 (parameters) = 19,200 SBUs on thin nodes.
- Boolean Fourier analysis. We performed preliminary studies on small lattices (24 spins) and found that one lattice is processed with 10 core-hours on our local node. The scaling of our algorithm is a bit super-exponential, so to perform the same study on 36-spin lattice we need 2 ** (36-24) × 3 (lattices) × 10 (core-hours) = 122,880 SBUs on himem node.
- Discrete optimization: for small systems (on 18 spins) we need 4 core-hours for one annealing optimization, according to our preliminary studies. For larger systems, we need an order of magnitude more compute. During each full optimization run, we need to perform the annealing at each inner loop step up, that takes about 50 to converge. Thus we need 3 (lattices) × 20 (parameters) × 40 (core-hours) × 50 (inner loop steps) = 120,000 SBUs on thin nodes.
- We reserve the rest 21,224 SBUs for trial and error and fine-tuning of our algorithms.

Memory

One of our core algorithms, Hadamard transform for Boolean Fourier analysis, needs to store full vectors in the memory. One ground state is about 512 GB, as well as its Fourier transform. We need to store several copies to make comparisons between predictions and ground state. Thus we need to run this code on 4TB node.

Storage

We require 10TB of project space as one ground state is about 512 GB and we need to keep cached ground states for various values of the parameters.

Software

We are planning mostly to use the Nix package manager to install and manage all software in our userspace. For working with neural networks, we will use the PyTorch framework, either from Nix or from the module environment.

#### Resources

- Resources: Snellius
- GPU Snellius: Yes
- SBU Snellius: 1000,000
- Terabyte Project Space Snellius: 10TB