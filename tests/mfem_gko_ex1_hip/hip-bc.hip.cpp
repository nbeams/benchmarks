#include "hip-bc.hpp"
  
// Minimal recreation of MFEM's "FORALL" wrapper as used for ConstrainedOperators: kernel,
// wrapper for calling kernel, function that chooses device or host.
template<typename T>
__launch_bounds__(256) __global__ static void HipSetConstraintsKernel(const int N, const int* constraint_list, T* d_data)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   d_data[constraint_list[k]] = 0.0;
}

template<typename T>
void HipWrapConstraintSet(const int N, const int* constraint_list, T *d_data)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   HipSetConstraintsKernel<<<GRID,BLCK>>>(N, constraint_list, d_data);
}
template void HipWrapConstraintSet(const int N, const int* constraint_list, double *d_data);
template void HipWrapConstraintSet(const int N, const int* constraint_list, float *d_data);

template<typename T>
__launch_bounds__(256) __global__ static void HipTransferMeshCoordsKernel(const int N, const double* mfem_coords, T* coords)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   coords[k] = (T) mfem_coords[k];
}

template<typename T>
void HipTransferMeshCoords(const int N, const double* mfem_coords, T* coords)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   HipTransferMeshCoordsKernel<T><<<GRID,BLCK>>>(N, mfem_coords, coords);
}

template void HipTransferMeshCoords<double>(const int N, const double* mfem_coords, double* coords);
template void HipTransferMeshCoords<float>(const int N, const double* mfem_coords, float* coords);
