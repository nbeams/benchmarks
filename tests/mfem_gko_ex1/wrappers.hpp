#include <ginkgo/ginkgo.hpp>
#include <ceed/ceed.h>
#include <mfem.hpp>

// These functions are taken from MFEM and modified as needed
// to meet the purposes of this test.
void InitRestriction(const mfem::FiniteElementSpace &fes,
                     Ceed ceed,
                     CeedElemRestriction *restr);

void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            Ceed ceed,
                            CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                            const CeedInt *strides,
                            CeedElemRestriction *restr);

template<typename ValueType>
void InitTensorBasis(const mfem::FiniteElementSpace &fes,
                            const mfem::FiniteElement &fe,
                            const mfem::IntegrationRule &ir,
                            Ceed ceed, CeedBasis *basis);
template<typename ValueType>
void InitNonTensorBasis(const mfem::FiniteElementSpace &fes,
                               const mfem::FiniteElement &fe,                                   
                               const mfem::IntegrationRule &ir, 
                               Ceed ceed, CeedBasis *basis);

/**
* This class wraps a CeedOperator for Ginkgo, to make its Apply()
* function available to Ginkgo, provided the input and output vectors
* are of the CeedVectorWrapper type.
* ValueType is the precision used for the data of the input/output vectors.
* @ingroup Ginkgo
*/
template<typename ValueType>
class CeedOperatorWrapper
   : public gko::EnableLinOp<CeedOperatorWrapper<ValueType>>,
     public gko::EnableCreateMethod<CeedOperatorWrapper<ValueType>>
{
public:
   CeedOperatorWrapper(std::shared_ptr<const gko::Executor> exec,
                       gko::size_type size = 0,
                       Ceed op_ceed = NULL,
                       CeedOperator op = NULL,
                       CeedVector interior = NULL)
      : gko::EnableLinOp<CeedOperatorWrapper>(exec, gko::dim<2> {size, size}),
   gko::EnableCreateMethod<CeedOperatorWrapper>()
   {
     oper = op;
     ceed = op_ceed;
     CeedVectorCreate(ceed, size, &ceed_x);
     CeedVectorCreate(ceed, size, &ceed_b);
     int_marker = &interior;

     CeedScalar *int_nodes;
     CeedVectorGetArray(*int_marker, CEED_MEM_HOST, &int_nodes);
     CeedSize sol_size;
     CeedVectorGetLength(*int_marker, &sol_size);
     int num_bd_nodes = 0;
     for (CeedSize i = 0; i < sol_size; i++) {
       if (int_nodes[i] < 1.e-5) {
          num_bd_nodes++;
       }
     }
     int *ess_bdr_nodes = (int*) malloc(sizeof(int)*num_bd_nodes);
     CeedInt node_ctr = 0;
     for (CeedSize i = 0; i < sol_size; i++) {
       if (int_nodes[i] < 1.e-5) {
        ess_bdr_nodes[node_ctr] = i;
        node_ctr++;
      }
     }
     CeedVectorRestoreArray(*int_marker, &int_nodes);
     num_constraints = num_bd_nodes;
     if (exec->get_master() != exec) {
        // Allocate memory on device
        int *ess_bdr_dev = exec->alloc<int>(
              static_cast<gko::size_type>(num_bd_nodes));     
        // Copy from host to device
        exec->copy_from(
            exec->get_master().get(), 
            static_cast<gko::size_type>(num_bd_nodes), 
            ess_bdr_nodes, ess_bdr_dev);
        ess_bdr = gko::Array<int>(exec, static_cast<gko::size_type>(num_bd_nodes), 
               ess_bdr_dev);

        free(ess_bdr_nodes);
     }
     else {
       ess_bdr = gko::Array<int>(exec, static_cast<gko::size_type>(num_bd_nodes), 
              static_cast<int*>(ess_bdr_nodes));
     }
   }
  
   virtual ~CeedOperatorWrapper()
   {
      CeedOperatorDestroy(&oper);
      CeedVectorDestroy(&ceed_x);
      CeedVectorDestroy(&ceed_b);
   }
   CeedOperator& GetCeedOperator() { return oper; }

protected:
   void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override;
   void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                   const gko::LinOp *beta, gko::LinOp *x) const override;

private:
   Ceed ceed;
   CeedOperator oper;
   CeedVector ceed_x, ceed_b;
   CeedVector* int_marker;
   gko::Array<int> ess_bdr;
   int num_constraints;
};

