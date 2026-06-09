Steps to build and run this benchmark
=====================================

(Assuming Ginkgo is already built)
Starting from the top-level benchmarks directory:

1. `cd machine-configs`

2. Edit the machine configuration in linux.sh as necessary to match the machine. E.g., check `cuda_arch`/`hip_arch`, `cuda_home`/`hip_home`, etc.
   Set `memory_per_node` (in GB) for the machine.
   Also edit `set_mpi_options()` to set the desired mpi executable options and flags for launching job steps on the machine.

3. `cd ../tests/mfem_ex1_gko`

4. In `ex1_gko.sh`, un-comment the relevant line at the end (CUDA or HIP build).

5. Build for CUDA: 
`GINKGO_DIR=/path/to/gko/install ../../go.sh -c linux -m gcc -r ex1_gko.sh build_only=1`

Or build for HIP:
`GINKGO_DIR=/path/to/gko/install ../../go.sh -c linux -m hip -r ex1_gko.sh build_only=1`

6. Run the benchmark:

Sample  CUDA run:

```
# Run diffusion test (problem 1), NUM_RANKS processes, with Ginkgo solver and matrix-free operator
../../go.sh -c linux -m gcc -r ex1_gko.sh -n NUM_RANKS --proc-node NUM_RANKS_PER_NODE 'problem=1' 'min_p=1' 'max_p=8' \
'mfem_devs="cuda"' 'solver=1' 'assemble=0' max_dofs_proc=16800000 > ex1.data`
```

Sample  HIP run:

```
# Run diffusion test (problem 1), NUM_RANKS processes, with Ginkgo solver and fully-assembled matrix operator
../../go.sh -c linux -m hip -r ex1_gko.sh -n NUM_RANKS --proc-node NUM_RANKS_PER_NODE 'problem=1' 'min_p=1' 'max_p=8' \
'mfem_devs="hip"' 'solver=1' 'assemble=1' max_dofs_proc=16800000 > ex1.data`
```

List of options:

problem=0     : mass operator test
problem=1     : diffusion operator test

solver=0      : MFEM CG solver
solver=1      : Ginkgo CG solver

assemble=0    : matrix-free operator
assemble=1    : assembled parallel matrix (HypreParMatrix)

max_dofs_proc : max DOFs per rank. The test will refine the mesh until reaching this size

min_p         : minimum order of basis functions used (tensor-product hexahedron)
max_p         : maximum order of basis functions used; the test will loop through all orders [min_p, max_p]
