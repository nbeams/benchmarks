// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project
// (17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
// organizations (Office of Science and the National Nuclear Security
// Administration) responsible for the planning and preparation of a capable
// exascale ecosystem, including software, applications, hardware, advanced
// system engineering and early testbed platforms, in support of the nation's
// exascale computing imperative.

//
//   CEED Bake-off PROBLEM, based on MFEM Example 1p
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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
   const char *device_config = "cpu";

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
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

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
      cout << "Total number of elements: " << global_ne << endl;
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   MFEM_VERIFY(order > 0, "invalid 'order': " << order);
   fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   long size = fespace->GetTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
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

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   if (problem == 0)
   {
      a->AddDomainIntegrator(new MassIntegrator(one));
   }
   else
   {
      a->AddDomainIntegrator(new DiffusionIntegrator(one));
   }

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use no preconditioner, for now.
   const int max_cg_iter = 200;
   const int cg_print_level = 3;
   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(max_cg_iter);
   cg.SetPrintLevel(cg_print_level);
   cg.SetOperator(*A);

   // Warm-up CG solve (in case of JIT to avoid timing it)
   {
      Vector Xtmp(X);
      cg.SetMaxIter(2);
      cg.SetPrintLevel(-1);
      cg.Mult(B, Xtmp);
      cg.SetMaxIter(max_cg_iter);
      cg.SetPrintLevel(cg_print_level);
   }
   
   // Start CG timing.
   tic_toc.Clear();
   
   // Start & Stop CG timing.
   tic_toc.Start();
   cg.Mult(B, X);
   tic_toc.Stop();
   double rt_min, rt_max, my_rt;
   my_rt = tic_toc.RealTime();
   rt_min = my_rt;
   rt_max = my_rt;
   
   // Print timing results.
   if (myid == 0)
   {
      int cg_iter = cg.GetNumIterations();
      // Note: In the pcg algorithm, the number of operator Mult() calls is
      //       N_iter and the number of preconditioner Mult() calls is N_iter+1.
      cout << '\n'
           << "Total CG time:    " << rt_max << " (" << rt_min << ") sec."
           << endl;
      cout << "Time per CG step: "
           << rt_max / cg_iter << " ("
           << rt_min / cg_iter << ") sec." << endl;
      cout << "\n\"DOFs/sec\" in CG: "
           << 1e-6*size*cg_iter/rt_max << " ("
           << 1e-6*size*cg_iter/rt_min << ") million.\n"
           << endl;
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

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
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

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
      cout << "Processor partitioning: ";
      nxyz.Print(cout, dim);

      // Mesh dimensions AFTER parallel refinement:
      cout << "Mesh dimensions: "
           << (1 << (t[0]+par_ref_levels)) << ' '
           << (1 << (t[1]+par_ref_levels)) << ' '
           << (1 << (t[2]+par_ref_levels)) << endl;
   }

   return mesh;
}
