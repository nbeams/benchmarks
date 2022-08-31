#include <ceed/ceed.h>
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <hip/hip_runtime.h>
#include "hip-bc.hpp"

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

//using namespace std;
using namespace mfem;

template<typename ValueType>
void TransferMeshCoords(Vector &mfem_coords, CeedVector ceed_coords, CeedMemType mem) 
{
   CeedScalar *mesh_coords_data;
   int size = mfem_coords.Size();
   CeedVectorGetArrayWrite(ceed_coords, mem, &mesh_coords_data);
   if (mem == CEED_MEM_DEVICE) {
     const double *mfem_mesh_coords_data = mfem_coords.Read(true);
     HipTransferMeshCoords<ValueType>(size, mfem_mesh_coords_data, mesh_coords_data);
   } 
   else {
     const double *mfem_mesh_coords_data = mfem_coords.Read(false);
     for(int i = 0; i < size; i++) { 
       mesh_coords_data[i] = mfem_mesh_coords_data[i];
     }
   }
   CeedVectorRestoreArray(ceed_coords, &mesh_coords_data);

}

// These functions are extracted from MFEM: fem/ceed/interface/restriction.cpp
static void InitNativeRestr(const mfem::FiniteElementSpace &fes,
                            Ceed ceed, CeedElemRestriction *restr)
{  
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   const mfem::Array<int>& dof_map = tfe->GetDofMap();
   
   for (int i = 0; i < fes.GetNE(); i++)
   {  
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {  
         tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
      }
   }
   
   CeedElemRestrictionCreate(ceed, fes.GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitLexicoRestr(const mfem::FiniteElementSpace &fes,
                            Ceed ceed, CeedElemRestriction *restr)
{  
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   
   for (int e = 0; e < fes.GetNE(); e++)
   {
      for (int i = 0; i < P; i++)
      {
         tp_el_dof[i + e*P] = stride*el_dof.GetJ()[i + e*P];
      }
   }

   CeedElemRestrictionCreate(ceed, fes.GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitRestriction(const FiniteElementSpace &fes,
                     Ceed ceed,
                     CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   if ( tfe && tfe->GetDofMap().Size()>0 ) // Native ordering using dof_map
   {
      InitNativeRestr(fes, ceed, restr);
   }
   else  // Lexicographic ordering
   {
      InitLexicoRestr(fes, ceed, restr);
   }
}

void InitStridedRestriction(const mfem::FiniteElementSpace &fes, Ceed ceed,
                            CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                            const CeedInt *strides,
                            CeedElemRestriction *restr)
{
   
   CeedElemRestrictionCreateStrided(ceed, nelem, nqpts, qdatasize,
                                    nelem*nqpts*qdatasize,
                                    strides,
                                    restr);
}

template<typename ValueType>
void InitTensorBasis(const mfem::FiniteElementSpace &fes,
                            const mfem::FiniteElement &fe,
                            const mfem::IntegrationRule &ir,
                            Ceed ceed, CeedBasis *basis)
{
   const mfem::DofToQuad &maps = fe.GetDofToQuad(ir, mfem::DofToQuad::TENSOR);
   mfem::Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();                                                           
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   // Here is where we differ from the MFEM version: 
   ValueType qX[nqpts];
   ValueType qW[nqpts];
   ValueType Bt[nqpts * ndofs];
   ValueType Gt[nqpts * ndofs * dim];
   const double *mfem_Bt = maps.Bt.GetData();
   const double *mfem_Gt = maps.Gt.GetData();
   // The x-coordinates of the first `nqpts` points of the integration rule are
   // the points of the corresponding 1D rule. We also scale the weights
   //  accordingly.

   double w_sum = 0.0;
   for (int i = 0; i < nqpts; i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qX[i] = (ValueType) ip.x;
      qW[i] = (ValueType) ip.weight;
      w_sum += ip.weight;
   }
   for (int i = 0; i < nqpts; i++) 
   {
     qW[i] = (ValueType) ((double) qW[i] / w_sum); 
   }
   for (int i = 0; i < nqpts * ndofs; i++)
   {
     Bt[i] = (ValueType) mfem_Bt[i];
   }
   for (int i = 0; i < nqpts * ndofs * dim; i++)
   {
     Gt[i] = (ValueType) mfem_Gt[i];
   }
   CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes.GetVDim(), ndofs,
                           nqpts, Bt,
                           Gt, qX,
                           qW, basis);
}

template<typename ValueType>
void InitNonTensorBasis(const mfem::FiniteElementSpace &fes,                             
                               const mfem::FiniteElement &fe,                                   
                               const mfem::IntegrationRule &ir,                                 
                               Ceed ceed, CeedBasis *basis)
{                                                                                               
   const mfem::DofToQuad &maps = fe.GetDofToQuad(ir, mfem::DofToQuad::FULL);                    
   mfem::Mesh *mesh = fes.GetMesh();                                                            
   const int dim = mesh->Dimension();                                                           
   const int ndofs = maps.ndof;                                                                 
   const int nqpts = maps.nqpt;                                                                 
   // Again, here is where we begin modifications to MFEM's code...
   ValueType qX[dim * nqpts];
   ValueType qW[nqpts];
   ValueType Bt[nqpts * ndofs];
   ValueType Gt[nqpts * ndofs * dim];
   const double *mfem_Bt = maps.Bt.GetData();
   const double *mfem_Gt = maps.Gt.GetData();
   for (int i = 0; i < nqpts; i++)                                                             
   {                                                                                    
      const mfem::IntegrationPoint &ip = ir.IntPoint(i); 
      qX[0 * nqpts + i] = (ValueType) ip.x;
      if (dim>1) { qX[1 * nqpts + i] = (ValueType) ip.y; } 
      if (dim>2) { qX[2 * nqpts + i] = (ValueType) ip.z; } 
      qW[i] = (ValueType) ip.weight;
   }
   for (int i = 0; i < nqpts * ndofs; i++)
   {
     Bt[i] = (ValueType) mfem_Bt[i];
   }
   for (int i = 0; i < nqpts * ndofs * dim; i++)
   {
     Gt[i] = (ValueType) mfem_Gt[i];
   }
   // TODO: error if dim = 1?
   if (dim == 2) { 
       CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE,
                         fes.GetVDim(), ndofs, nqpts,
                         Bt, Gt,
                         qX, qW, basis);                                        
   }
   else if (dim == 3) { 
       CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TET,
                         fes.GetVDim(), ndofs, nqpts,
                         Bt, Gt,
                         qX, qW, basis);                                        
   }
}                                                              

struct BuildContext { CeedInt dim, space_dim; };

Mesh *make_mesh(int myid, int num_procs, int dim, int level,
                int &par_ref_levels, Array<int> &nxyz, int el_type);

int main(int argc, char *argv[])
{
   int num_procs = 1;
   int myid = 0;

   // 2. Parse command-line options.
   int dim = 3;
   int level = 0;
   int order = 1;
   int problem = 0;
   int el_type = 0;
   int output_to_file = 0;
   const char *device_config = "cpu";
   const char *ceed_spec = "/cpu/self/ref/serial";

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--mesh-dimension",
                  "Solve 2D or 3D problem.");
   args.AddOption(&level, "-l", "--refinement-level",
                  "Set the problem size: 2^level mesh elements per processor.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&problem, "-p", "--problem", "Problem 0:Mass, 1:Diffusion.");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type 0:Hexahedron, 1:Tetrahedron.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ceed_spec, "-c", "--ceed",
                  "libCEED backend configuration string.");
   args.AddOption(&output_to_file, "-out", "--output",
                  "Whether to print the solution vector.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(std::cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Select appropriate backend and logical device based on the <ceed-spec>
   // command line argument -- unless it doesn't match with the MFEM device 
   // config -- in which case, pick a backend that will work with MFEM and 
   // print a warning.
   Ceed ceed;
   const char *ceed_device = strstr(ceed_spec, "gpu");
   if (ceed_device) { // libCEED spec is for a GPU backend
     if (device.Allows(Backend::CUDA_MASK)) {
       const char *ceed_cuda = strstr(ceed_spec, "cuda");
       if (ceed_cuda) {
         CeedInit(ceed_spec, &ceed);
         std::cout << "ceed_name: " << ceed_spec << "\n";
       }
       else {
         CeedInit("/cpu/self/ref/serial", &ceed);
         std::cout << "WARNING!!!! MFEM/libCEED configuration mismatch.  Using cpu-ref backend!\n";
       }
     }
     else if (device.Allows(Backend::HIP_MASK)) {
       const char *ceed_hip = strstr(ceed_spec, "hip");
       if (ceed_hip) {
         CeedInit(ceed_spec, &ceed);
         std::cout << "ceed_name: " << ceed_spec << "\n";
       }
       else {
         CeedInit("/cpu/self/ref/serial", &ceed);
         std::cout << "WARNING!!!! MFEM/libCEED configuration mismatch.  Using cpu-ref backend!\n";
       }
     }
     else {
       CeedInit("/cpu/self/ref/serial", &ceed);
       std::cout << "WARNING!!!! MFEM/libCEED configuration mismatch.  Using cpu-ref backend!\n";
     }
   } 
   else { // libCEED spec is for a CPU backend
     if (device.Allows(Backend::CUDA_MASK)) {
       CeedInit("/gpu/cuda/ref", &ceed);
       std::cout << "WARNING!!!! MFEM/libCEED configuration mismatch.  Using cuda-ref backend!\n";
     }
     else if (device.Allows(Backend::HIP_MASK)) {
       CeedInit("/gpu/hip/ref", &ceed);
       std::cout << "WARNING!!!! MFEM/libCEED configuration mismatch.  Using hip-ref backend!\n";
     }
     else {
       CeedInit(ceed_spec, &ceed);
       std::cout << "ceed_name: " << ceed_spec << "\n";
     }
   }
   CeedMemType mem;
   CeedGetPreferredMemType(ceed, &mem); 

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   int par_ref_levels;
   Array<int> nxyz;
   Mesh *mesh = make_mesh(myid, num_procs, dim, level, par_ref_levels, nxyz, el_type);

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      // skip
   }
   for (int l = 0; l < par_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   // mesh->PrintInfo();
   long global_ne = mesh->GetNE();
   if (myid == 0)
   {
      std::cout << "Total number of elements: " << global_ne << std::endl;
   }
   CeedInt nelem = global_ne;  // because this is serial...

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   MFEM_VERIFY(order > 0, "invalid 'order': " << order);
   fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   mesh->EnsureNodes();
   const FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   CeedInt sol_size = fespace->GetTrueVSize();
   CeedInt mesh_size = mesh_fes->GetTrueVSize();
   if (myid == 0)
   {
      std::cout << "Number of finite element unknowns: " << sol_size << std::endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   // This vector is used by the OperWrapper to set which nodes are interior nodes.
   // It is also used to zero out the rows associatd with the homogenous Dirichlet BCs
   // in the RHS vector (through libCEED's PointwiseMult).
   CeedVector int_nodes;
   CeedVectorCreate(ceed, sol_size, &int_nodes);
   CeedVectorSetValue(int_nodes, 1.0);
   CeedScalar *int_nodes_data;
   CeedVectorGetArray(int_nodes, CEED_MEM_HOST, &int_nodes_data);
   for (int i = 0; i < ess_tdof_list.Size(); i++) 
   {
      int_nodes_data[ess_tdof_list[i]] = 0.0;   
   }
   CeedVectorRestoreArray(int_nodes, &int_nodes_data);

   const mfem::FiniteElement &fe = *fespace->GetFE(0);
   const mfem::FiniteElement &mesh_fe = *mesh_fes->GetFE(0);
   std::cout << "ndofs, mesh_fe: " << mesh_fe.GetDof() << "\n";
   std::cout << "ndofs, fe: " << fe.GetDof() << "\n";
   ConstantCoefficient one(1.0);
   DiffusionIntegrator *d_integ = new DiffusionIntegrator(one); // Make an integrator
   const IntegrationRule *ir = &(d_integ->GetRule(fe, fe));  // Get the integration rule -- we will pass this to libCEED 
                                                             // for setting up the Basis object
   MassIntegrator *m_integ = new MassIntegrator(one); // Repeat for mass integrator
   const IntegrationRule *ir_mass = &(m_integ->GetRule(fe, fe, *fespace->GetElementTransformation(0)));
   if (problem == 0) // mass problem, use mass IR for both Basis objects
     ir = ir_mass;
 
   // Build CeedElemRestriction objects describing the mesh and solution discrete
   // representations, and mesh and solution bases.
   CeedBasis mesh_basis, sol_basis;
   if (el_type == 0) // hex elements
   {
     InitTensorBasis<CeedScalar>(*mesh_fes, mesh_fe, *ir, ceed, &mesh_basis);
     InitTensorBasis<CeedScalar>(*fespace, fe, *ir, ceed, &sol_basis);
   }
   else // non-tensor (tet)
   {
     InitNonTensorBasis<CeedScalar>(*mesh_fes, mesh_fe, *ir, ceed, &mesh_basis);
     InitNonTensorBasis<CeedScalar>(*fespace, fe, *ir, ceed, &sol_basis);
   } 
   delete d_integ;
   delete m_integ;

   CeedInt elem_nqpts;
   CeedBasisGetNumQuadraturePoints(sol_basis, &elem_nqpts);
   std::cout << "nqpts: " << elem_nqpts << "\n";
 
   CeedElemRestriction mesh_restr, sol_restr, q_data_restr_i;
   const int qdatasize = (problem == 0) ? 1 : dim*(dim+1)/2;
   InitStridedRestriction(*mesh_fes, ceed, nelem, elem_nqpts, qdatasize,
                                CEED_STRIDES_BACKEND,
                                &q_data_restr_i);
   InitRestriction(*mesh_fes, ceed, &mesh_restr);
   InitRestriction(*fespace, ceed, &sol_restr);

   // Create a CeedVector with the mesh coordinates.
   CeedVector mesh_coords;
   CeedVectorCreate(ceed, mesh_size, &mesh_coords);
   Vector mfem_mesh_coords;
   mesh->GetNodes(mfem_mesh_coords);
   TransferMeshCoords<CeedScalar>(mfem_mesh_coords, mesh_coords, mem);

   // Context data to be passed to the 'f_build_diff' QFunction.
   CeedQFunctionContext build_ctx;
   struct BuildContext build_ctx_data;
   build_ctx_data.dim = build_ctx_data.space_dim = dim;
   CeedQFunctionContextCreate(ceed, &build_ctx);
   CeedQFunctionContextSetData(build_ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof(build_ctx_data), &build_ctx_data); 
 
   // Create the QFunction that builds the main operator (i.e. computes its
   // quadrature data) and set its context data.
   CeedQFunction qf_build;   
   if (problem == 0) {
     char qf_build_name[13] = "";
     snprintf(qf_build_name, sizeof(qf_build_name), "Mass%dDBuild", dim);
     CeedQFunctionCreateInteriorByName(ceed, qf_build_name, &qf_build);
   } else { 
     char qf_build_name[16] = "";
     snprintf(qf_build_name, sizeof(qf_build_name), "Poisson%dDBuild", dim);
     CeedQFunctionCreateInteriorByName(ceed, qf_build_name, &qf_build);
   }

   // Create the operator that builds the quadrature data for the main problem 
   // operator.
   CeedOperator op_build;
   CeedOperatorCreate(ceed, qf_build, CEED_QFUNCTION_NONE,
                      CEED_QFUNCTION_NONE, &op_build);
   CeedOperatorSetField(op_build, "dx", mesh_restr, mesh_basis,
                        CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(op_build, "weights", CEED_ELEMRESTRICTION_NONE,
                        mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(op_build, "qdata", q_data_restr_i,
                        CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

   // Compute the quadrature data for the operator.
   CeedVector q_data;
   CeedVectorCreate(ceed, nelem*elem_nqpts*qdatasize, &q_data);
   CeedOperatorApply(op_build, mesh_coords, q_data,
                     CEED_REQUEST_IMMEDIATE);

   // Create the QFunction that defines the action of the main operator.
   CeedQFunction qf_apply;
   if (problem == 0) {
     CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_apply);
   } else {
     char qf_apply_name[16] = "";
     snprintf(qf_apply_name, sizeof(qf_apply_name), "Poisson%dDApply", dim);
     CeedQFunctionCreateInteriorByName(ceed, qf_apply_name, &qf_apply);
   }

   // Create the main operator.
   CeedOperator op_apply;
   CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE,
                      CEED_QFUNCTION_NONE, &op_apply);
   if (problem == 0) {
     CeedOperatorSetField(op_apply, "u", sol_restr, sol_basis, CEED_VECTOR_ACTIVE);
   } else {
     CeedOperatorSetField(op_apply, "du", sol_restr, sol_basis, CEED_VECTOR_ACTIVE);
   }
   CeedOperatorSetField(op_apply, "qdata", q_data_restr_i, CEED_BASIS_COLLOCATED,
                        q_data);
   if (problem == 0) {
     CeedOperatorSetField(op_apply, "v", sol_restr, sol_basis, CEED_VECTOR_ACTIVE);
   } else {
     CeedOperatorSetField(op_apply, "dv", sol_restr, sol_basis, CEED_VECTOR_ACTIVE);
   }

   // Create auxiliary solution-size vectors.
   CeedVector u, v;
   CeedVectorCreate(ceed, sol_size, &u);
   CeedVectorCreate(ceed, sol_size, &v);

   // Initialize 'u' with ones.
   CeedScalar *u_array;
   CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
   for (CeedInt i = 0; i < sol_size; i++) {
     u_array[i] = 1.0;
   }
   CeedVectorRestoreArray(u, &u_array);

   if (output_to_file) {
     //TMP
     CeedVector ones;
     CeedVectorCreate(ceed, sol_size, &ones);
     CeedScalar *ones_array;
     CeedVectorGetArrayWrite(ones, CEED_MEM_HOST, &ones_array);
     for (CeedInt i = 0; i < sol_size; i++) {
       ones_array[i] = 1.0;
     }
     CeedVectorRestoreArray(ones, &ones_array);
     CeedVector out_ones;
     CeedVectorCreate(ceed, sol_size, &out_ones);
     CeedOperatorApply(op_apply, ones, out_ones, CEED_REQUEST_IMMEDIATE); 
     const CeedScalar *data;
     CeedVectorGetArrayRead(out_ones, CEED_MEM_HOST, &data);
     std::ofstream output_file;
     output_file.open("ones-output.dat");
     for (CeedInt i = 0; i < sol_size; i++) {
        output_file << std::setprecision(16) << std::scientific << data[i] << "\n";
     }
     output_file.close();
     CeedVectorRestoreArrayRead(out_ones, &data);
     CeedVectorDestroy(&ones);
     CeedVectorDestroy(&out_ones);
   }


   // Testing: warm-up apply
   CeedVectorSetValue(v, 0.0);
   CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);
   // Set x back to zero
   CeedVectorSetValue(v, 0.0);

   // Start CG timing.
   tic_toc.Clear();
   
   // Start & Stop CG timing.
   int num_app = 2000;
   tic_toc.Start();
   for (int i = 0; i < num_app; i++) {
     CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);
   }
   hipDeviceSynchronize();
   tic_toc.Stop();
   double my_rt;
   my_rt = tic_toc.RealTime();
   
   // Print timing results.
   if (myid == 0)
   {
      std::cout << "Total mults: " << num_app << "\n";
      // Note: In the pcg algorithm, the number of operator Mult() calls is
      //       N_iter and the number of preconditioner Mult() calls is N_iter+1.
      std::cout << '\n'
           << "Total mult time:    " << my_rt << " sec."
           << std::endl;
      std::cout << "Time per mult: "
           << my_rt / num_app << " sec." << std::endl;
      std::cout << "\n\"DOFs/sec\" in mult: "
           << 1e-6*sol_size*num_app/my_rt << " million.\n";
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      // skip
   }

   // 16. Send the solution by socket to a GLVis server.
   // if (visualization)
   {
      // skip
   }

   // 17. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;
   CeedVectorDestroy(&u);
   CeedVectorDestroy(&v);
   CeedVectorDestroy(&mesh_coords);
   CeedVectorDestroy(&int_nodes);
   CeedVectorDestroy(&q_data);
   CeedVectorDestroy(&mesh_coords);
   CeedOperatorDestroy(&op_apply);
   CeedQFunctionDestroy(&qf_apply);
   CeedQFunctionContextDestroy(&build_ctx);
   CeedOperatorDestroy(&op_build);
   CeedQFunctionDestroy(&qf_build);
   CeedElemRestrictionDestroy(&sol_restr);
   CeedElemRestrictionDestroy(&mesh_restr);
   CeedElemRestrictionDestroy(&q_data_restr_i);
   CeedBasisDestroy(&sol_basis);
   CeedBasisDestroy(&mesh_basis);
   CeedDestroy(&ceed);

   return 0;
}

Mesh *make_mesh(int myid, int num_procs, int dim, int level,
                int &par_ref_levels, Array<int> &nxyz, int el_type)
{
   int log_p = (int)floor(log((double)num_procs)/log(2.0) + 0.5);
   MFEM_VERIFY((1 << log_p) == num_procs,
               "number of processor is not a power of 2: " << num_procs);
   MFEM_VERIFY(dim == 3, "dim = " << dim << " is NOT implemented!");

   // Determine processor decomposition.
   int s[3];
   s[0] = log_p/3 + (log_p%3 > 0 ? 1 : 0);
   s[1] = log_p/3 + (log_p%3 > 1 ? 1 : 0);
   s[2] = log_p/3;
   nxyz.SetSize(dim);
   nxyz[0] = 1 << s[0];
   nxyz[1] = 1 << s[1];
   nxyz[2] = 1 << s[2];

   // Determine mesh size.
   int ser_level = level%3;
   par_ref_levels = level/3;
   int log_n = log_p + ser_level;
   int t[3];
   t[0] = log_n/3 + (log_n%3 > 0 ? 1 : 0);
   t[1] = log_n/3 + (log_n%3 > 1 ? 1 : 0);
   t[2] = log_n/3;

   // Create the Mesh.
   const bool sfc_ordering = true;
   Mesh *mesh = NULL;
  if (el_type == 0)
  {  // Hex elements
    mesh = new Mesh(Mesh::MakeCartesian3D(1 << t[0], 1 << t[1], 1 << t[2],
                                          Element::HEXAHEDRON,
                                          1.0, 1.0, 1.0, sfc_ordering));
   }
   else
   { // Tets
    mesh = new Mesh(Mesh::MakeCartesian3D(1 << t[0], 1 << t[1], 1 << t[2],
                                          Element::TETRAHEDRON,
                                          1.0, 1.0, 1.0, sfc_ordering));
   }
   if (myid == 0)
   {
      std::cout << "Processor partitioning: ";
      nxyz.Print(std::cout, dim);

      // Mesh dimensions AFTER parallel refinement:
      std::cout << "Mesh dimensions: "
           << (1 << (t[0]+par_ref_levels)) << ' '
           << (1 << (t[1]+par_ref_levels)) << ' '
           << (1 << (t[2]+par_ref_levels)) << std::endl;
   }

   return mesh;
}
