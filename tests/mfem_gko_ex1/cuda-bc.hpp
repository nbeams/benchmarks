#include <cuda.h>

#define BLCK 256

template<typename T>
void CuWrapConstraintSet(const int N, const int* constraint_list, T* d_data);

template<typename T>
void CuTransferMeshCoords(const int N, const double* mfem_coords, T* coords);
