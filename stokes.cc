/* ---------------------------------------------------------------------
 *  *
 *   *
 *    * ---------------------------------------------------------------------
 *
 *     */
#include <deal.II/base/logstream.h>               // Log system during compile/run
#include <deal.II/base/quadrature_lib.h>          // Quadrature Rules
#include <deal.II/base/convergence_table.h>       // Convergence Table
#include <deal.II/base/timer.h>                   // Timer class

#include <deal.II/dofs/dof_handler.h>             // DoF (defrees of freedom) Handler
#include <deal.II/dofs/dof_tools.h>               // Tools for Working with DoFs

#include <deal.II/fe/fe_q.h>                      // Continuous Finite Element Q Basis Function
#include <deal.II/fe/fe_values.h>                 // Values of Finite Elements on Cell
#include <deal.II/fe/fe_system.h>                 // System (vector) of Finite Elements

#include <deal.II/grid/tria.h>                    // Triangulation declaration
#include <deal.II/grid/tria_accessor.h>           // Access the cells of Triangulation
#include <deal.II/grid/tria_iterator.h>           // Iterate over cells of Triangulations
#include <deal.II/grid/grid_generator.h>          // Generate Standard Grids
#include <deal.II/grid/grid_out.h>                // Output Grids

#include <deal.II/lac/affine_constraints.h>       // Constraint Matrix
#include <deal.II/lac/dynamic_sparsity_pattern.h> // Dynamic Sparsity Pattern
#include <deal.II/lac/sparse_matrix.h>            // Sparse Matrix
#include <deal.II/lac/sparse_direct.h>            // UMFPACK

#include <deal.II/numerics/vector_tools.h>        // Interpolate Boundary Values
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>  // C++ Output
#include <fstream>   // C++ Output
#include <cmath>     // C++ Math Functions


using namespace dealii;

template <int dim>
class StokesProblem
{
  public:
    StokesProblem (const unsigned int degree);
    void run ();

    private:
      void setup_dofs ();
      void refine_grid();
      void assemble_stokes_system ();
      void stokes_solve ();
      void solution_update ();
      void setup_convergence_table();
      void output_results (const unsigned int cycle);
      void compute_errors (const unsigned int cycle);

      const unsigned int  degree;

      Triangulation<dim>      triangulation;
      FESystem<dim>           fe;
      DoFHandler<dim>         dof_handler;

      SparsityPattern         sparsity_pattern;
      SparseMatrix<double>    system_matrix;

      Vector<double>          solution;
      Vector<double>          delta_solution;
      Vector<double>          system_rhs;

      ConvergenceTable        convergence_table;

      ConstraintMatrix        zero_press_node_constraint;
};

// <<<<<<< Constructor

template<int dim>
StokesProblem<dim>::StokesProblem(const unsigned int degree)
    :
    degree(degree),
    fe(FE_Q<dim>(degree+1), dim,
    FE_Q<dim>(degree), 1), 
    dof_handler(triangulation)
    {}

template<int dim>
void StokesProblem<dim>::refine_grid()
{
  std::cout << "============================================================"
            << std::endl
            << "Globally refining domain..."
            << std::endl
            << "-------------------------------------------------------------"
            << std::endl;

  triangulation.refine_global(1);

  std::cout << "-------------------------------------------------------------"
            << std::endl
            << "Completed Globally refining domain..."
            << std::endl
            << "============================================================"
            << std::endl;
// >>>>>>> master
}

template <int dim>
void StokesProblem<dim>::run()
{
  // number of global refinements to peform on domain [0,1] with initially 4 cells
  // we increment to test convergence rates
  const unsigned int n_refinements = 3;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " Setting up domain and mesh..." << std::endl;
  make_grid();
  std::cout << " Completed Setting up domain and mesh..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Setting up boundary identifies..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  set_boundary();
  std::cout << " Completed Setting up boundary identifiers..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Refining the mesh globally..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (unsigned int i=0; i<n_refinements; ++i)
   refine_grid();
  std::cout << " Completed Refining the mesh globally..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  setup_dofs();
  std::cout << " Completed Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  set_pressure_constraint();
  std::cout << " Completed Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Setting up system pattern and size..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  setup_system();
  std::cout << " Completed Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Assembling system matrix..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  assemble_system();
  std::cout << " Completed Assembling system matrix..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Solving system..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  stokes_solve();
  std::cout << " Completed Solving system..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Computing errors..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  compute_errors();
  std::cout << " Completed Computing errors..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
 

}

template<int dim>
void StokesProblem<dim>::stokes_solve()
{
  SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);

  A_direct.vmult (solution, system_rhs);

  //solution = 0;  // testing Nav-Stokes with initial guess of zero
}
