#include <hip/hip_runtime.h>

#define BLCK 256

template<typename T>
void HipWrapConstraintSet(const int N, const int* constraint_list, T* d_data);

template<typename T>
void HipTransferMeshCoords(const int N, const double* mfem_coords, T* coords);
