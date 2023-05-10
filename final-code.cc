/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2021 by the deal.II authors
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

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/function_lib.h>

using namespace dealii;
 
 
 
template <int dim>
class Step6
{
public:
  Step6(const double alpha);
 
  void run(const unsigned int cycle_max);
 
private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;
 
  Triangulation<dim> triangulation;
 
  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;
 
  AffineConstraints<double> constraints;
 
  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;
 
  Vector<double> solution;
  Vector<double> system_rhs;

  const double alpha;
};

 


template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> & /*p*/,
                                const unsigned int /*component*/) const
{
return 0;
}
 
 
 
 
 
template <int dim>
Step6<dim>::Step6(double alpha)
  : fe(1)                         // degree of finite element is 1, i.e. linear.  can increase but will run slower
  , dof_handler(triangulation),
  alpha(alpha)
{}
 
 
 
 
template <int dim>
void Step6<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
 
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
 
 
  //VectorTools::interpolate_boundary_values(dof_handler,
  //                                         0,
  //                                         Functions::ZeroFunction<dim>(),
  //                                         constraints);
 
  constraints.close();
 
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
 
  sparsity_pattern.copy_from(dsp);
 
  system_matrix.reinit(sparsity_pattern);
}
 
 
 
template <int dim>
void Step6<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  const unsigned int n_face_q_points = face_quadrature_formula.size();

  RightHandSide<dim> right_hand_side;
 
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe,
                         face_quadrature_formula,
                         update_values | update_quadrature_points |
                         update_normal_vectors | update_JxW_values);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
 
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  
      for (const auto &cell : dof_handler.active_cell_iterators())
        {

          const double D = 1.0;
          const double neumann_value1 = 10.0;     // normal derivative at boundary 1
          // const double neumann_value2 = 1.0;
          // const double neumann_value3 = 1.0;
          // const double neumann_value4 = 1.0;



          cell_matrix = 0;
          cell_rhs    = 0;
    
          fe_values.reinit(cell);
    
          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                {
                  for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) +=
                      (D *                                  // D
                      fe_values.shape_grad(i, q_index) *    // grad phi_i(x_q)
                      fe_values.shape_grad(j, q_index) *    // grad phi_j(x_q)
                      fe_values.JxW(q_index))               // dx
                      -
                      (alpha) *                            // alpha
                      fe_values.shape_value(i,q_index)*     // phi_i(x_q)
                      fe_values.shape_value(j,q_index)*     // phi_j(x_q)
                      fe_values.JxW(q_index)                // dx
                      ;

                  const auto &x_q = fe_values.quadrature_point(q_index);
                  cell_rhs(i) += (right_hand_side.value(x_q) *            // f(x)
                                  fe_values.shape_value(i, q_index) *     // phi_i(x_q)
                                  fe_values.JxW(q_index));                // dx
                }
            }
          
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
              {
                if(face->boundary_id() == 1){
                fe_face_values.reinit(cell, face);

                for (unsigned int q_point = 0; q_point < n_face_q_points;
                      ++q_point)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) +=
                        (fe_face_values.shape_value(i, q_point) *   // phi_i(x_q)
                          neumann_value1 *                          // g(x_q)
                          fe_face_values.JxW(q_point));             // dx
                  }
                }
              }
    
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        }

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                0,
                                                RightHandSide<dim>(),
                                                boundary_values);
        MatrixTools::apply_boundary_values(boundary_values,
                                          system_matrix,
                                          solution,
                                          system_rhs);
    
}
 
 
 
 
template <int dim>
void Step6<dim>::solve()
{
  //SolverControl            solver_control(100000, 1e-6 * system_rhs.l2_norm());

  Timer timer;
  SparseDirectUMFPACK direct_solver;
  direct_solver.initialize(system_matrix);
  direct_solver.vmult(solution, system_rhs);
  timer.stop();

  // I used a direct solver because CG and GMRES were not giving a solution at every value of alpha

  std::cout << "   Time to solve: " << timer.cpu_time() << " s" << std::endl;

  //SolverGMRES<Vector<double>> solver(solver_control);
  //SolverCG<Vector<double>> solver(solver_control);
 
  // PreconditionSSOR<SparseMatrix<double>> preconditioner;
  // preconditioner.initialize(system_matrix, 1.2);
 
  // solver.solve(system_matrix, solution, system_rhs, preconditioner);
 
  constraints.distribute(solution);
}
 
 
 
template <int dim>
void Step6<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
 
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);
 
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);
 
  triangulation.execute_coarsening_and_refinement();
}
 
 
 
template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
  // {
  //   GridOut               grid_out;
  //   std::ofstream         output("grid-" + std::to_string(cycle) + ".gnuplot");
  //   GridOutFlags::Gnuplot gnuplot_flags(false, 5);
  //   grid_out.set_flags(gnuplot_flags);
  //   MappingQ<dim> mapping(3);
  //   grid_out.write_gnuplot(triangulation, output, &mapping);
  // }
 
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
 
    std::ofstream output("solution-alpha-" + std::to_string(alpha) + "-cycle-" + std::to_string(cycle)+ ".vtu");
    data_out.write_vtu(output);
  }


  Vector<float> difference (triangulation.n_active_cells());
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  VectorTools::integrate_difference (MappingQ<dim>(1),
                                     dof_handler,
                                     solution,
                                     Functions::ZeroFunction<dim>(),
                                     difference,
                                     quadrature_formula, 
                                     VectorTools::L2_norm);

    std::cout << "GREP" << cycle << ": " << alpha << " " << difference.l2_norm() << std::endl;    // output the l2 norm of the solution for each value of alpha and each cycle
}
 
 
 
template <int dim>
void Step6<dim>::run(const unsigned int cycle_max)
{
  //const unsigned int cycle_max = 5;


  std::cout << "   alpha: " << alpha
            << std::endl;

  for (unsigned int cycle = 0; cycle <= cycle_max; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;
 
      if (cycle == 0)
        {
          const double leftedge= 0;                   // "left" boundary of each side of hypercube
          const double rightedge = 1;                 // "right" boundary of each side of hypercube
          GridGenerator::hyper_cube(triangulation,leftedge,rightedge);
          for (const auto &cell : triangulation.active_cell_iterators())
                for (const auto &face : cell->face_iterators())     // loop to set boundary ids on each face of cube
                    if (face->center()[0] == leftedge)      // set boundary id on x=0
                      face->set_boundary_id(1);   
                    else if (face->center()[0]==rightedge)  // set boundary id on x=1
                      face->set_boundary_id(1);
                    else if (face->center()[1]==leftedge)   // set boundary id on y=0
                      face->set_boundary_id(1);
                    else if (face->center()[1]==rightedge)  // set boundary id on y=1
                      face->set_boundary_id(1);
                    else if (face->center()[2]==leftedge)   // set boundary id on z=0
                      face->set_boundary_id(1);
                    else if (face->center()[2]==rightedge)  // set boundary id on z=1
                      face->set_boundary_id(1);

          triangulation.refine_global(1);
        }
      else
        refine_grid();


 
 
      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;
 
      setup_system();
 
      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;
 
      assemble_system();
      
      solve();
      
      if(true/*cycle == cycle_max*/){
        output_results(cycle);
      }
    }
}
 
 
 
int main()
{
  try
    {
      double alpha_min = 1272.5;
      double alpha_max = 2000.;
      double alpha;               // -mu+nu
      double divisions = 1455;
      const unsigned int cycle_max = 7;

      std::cout<< "alpha_min=" << alpha_min << ",alpha_max=" << alpha_max << ",divisions=" << divisions << ",cycle_max=" << cycle_max << std::endl;

      for(double i=0. ; i<divisions+1. ; ++i){
        alpha = (alpha_max-alpha_min)*(i/divisions) + alpha_min;

        // try {

          Step6<3> reactor_problem(alpha);

          reactor_problem.run(cycle_max);
        // }
        // catch (...)
        //   {
        //     std::cout<<"Computations for alpha = " << alpha << " failed." << std::endl;
        //   }
      }
      
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
 
  return 0;
}
