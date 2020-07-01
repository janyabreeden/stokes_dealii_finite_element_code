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
      void assemble_navier_stokes_system ();
      void stokes_solve ();
      void navier_stokes_solve ();
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
