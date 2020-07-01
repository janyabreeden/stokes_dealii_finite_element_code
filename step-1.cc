/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 */

// @sect3{Include files}

#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h> //extra

#include <iostream>
#include <fstream>
#include <cmath>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h> //extra

#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
//extra
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

using namespace dealii;


class StokesProblem 
{
	public:
		StokesProblem(const unsigned int degree);
		~StokesProblem();
		void run();
	private:
		void setup_dofs();
		void refine_grid();
		void asemble_stokes_problem();
		void solve();
		void solution_update();
		void setup_convergence_table();
		void output_results();
		void compute_errors();

		const unsigned int degree;
		
         	 Triangulation<dim> triangulation;
		 DoFHandler<dim>    dof_handler;    
		 FESystem<dim>      fe;

		SparsityPattern spartsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> system_rhs;

		ConvergenceTable convergence_table;
		ConstraintMatrix zero_press_node_constraint;
};
template<int dim>
void StokesProblem<dim>::run()
{
SparseDirectUMFPACK  A_direct;
  A_direct.initialize(system_matrix);

  A_direct.vmult (solution, system_rhs);

  //solution = 0;  // testing Nav-Stokes with initial guess of zero
}
// @sect3{The main function}

int main()
{
try
{
	deallog.depth_console(0);
      StokesProblem<2> StokesProblem();
      flow_problem.run();
}
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;}
