#!/usr/bin/env python3
"""
Ray-tracing subproblem for contact detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import fem
import ufl
import basix
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc


def ray_tracing_sliding(
    parent_mesh, 
    contact_mesh, 
    facet_marker, 
    c_tag, 
    metadata, 
    nx,              # UFL normal expression on parent mesh
    x,               # UFL current position (X + u) on parent mesh
    xi_guess,        # Previous solution as initial guess (fem.Function or None)
    map_to_parent,   # Entity map for submesh assembly
    h0,              # Initial gap parameter
    d_val,           # Current indentation depth
    R,               # Indenter radius
    step_idx,        # Current step index for filenames
    output_dir,      # Output directory for plots
    y_cap=None,      # Height cap for parabola (flat region beyond cutoff)
    debug_plots=True,
    debug_monitor=False,
    u_field=None,
    # === ROBUSTNESS PARAMETERS ===
    reg_eps=0.0,         # Regularization strength (increased default)
):
    """
    Ray-tracing projection strategy.
    Nonlinear equation solved with Newton's method.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    ndim = parent_mesh.geometry.dim
    element_deg = 1  # Linear elements for xi
       
    # Function space on submesh
    V_xi = fem.functionspace(contact_mesh, ("CG", element_deg))
    
    xi = fem.Function(V_xi, name="xi")
    xi_old = fem.Function(V_xi, name="xi_old")
    
    # Get DOF coordinates
    dof_coords = V_xi.tabulate_dof_coordinates()
    X_dof = dof_coords[:, 0]
    
    # Store initial guess for fallback
    if xi_guess is not None:
        xi_fallback = xi_guess.x.array.copy()
    else:
        xi_fallback = X_dof 
    
    # Store last good solution for recovery from collapsed states
    xi_last_good = xi_fallback.copy()
    xi_expected_range = X_dof.max() - X_dof.min()
    
    converged = False
    final_reason = -1
            
    converged, reason = _solve_ray_tracing(
        xi, xi_old, V_xi, X_dof,
        parent_mesh, contact_mesh, facet_marker, c_tag, metadata,
        nx, x, xi_guess, map_to_parent,
        h0, d_val, R, y_cap,
        reg_eps=reg_eps,
        debug_monitor=debug_monitor,
        step_idx=step_idx,
    )
    
    final_reason = reason
        
    # CRITICAL: Detect and reject collapsed solutions
    xi_range = xi.x.array.max() - xi.x.array.min()
    if xi_range < 0.1 * xi_expected_range:
        # Solution collapsed - reject and restore from last good solution
        print(f"  WARNING: xi collapsed (range={xi_range:.4f}, expected ~{xi_expected_range:.4f}), rejecting solution")
        xi.x.array[:] = xi_last_good
        xi.x.scatter_forward()
        reason = -99  # Special code for collapsed solution
        final_reason = reason
        converged = False
    elif converged or reason >= -6:
        # Solution is reasonable - save as last good
        xi_last_good = xi.x.array.copy()
    
    # Store for plotting
    if xi_guess is not None:
        xi_old.x.array[:] = xi_guess.x.array[:]
    else:
        xi_old.x.array[:] = X_dof 
    xi_old.x.scatter_forward()
    
    # Generate diagnostic plots
    if debug_plots:
        try:
            _plot_geometry_debug_sliding(
                xi_old, xi, h0, d_val, R, y_cap, step_idx, output_dir, ndim,
                u_field=u_field  # Pass displacement field for current config plots
            )
        except Exception as e:
            print(f"  Warning: Could not generate diagnostic plots: {e}")
    
    if not converged:
        print(f"\n  WARNING: Ray-tracing projection did not converge!")
        print(f"  Reason code: {final_reason}")
    
    return xi


def _solve_ray_tracing(xi, xi_old, V_xi, X_dof,
                       parent_mesh, contact_mesh, facet_marker, c_tag, metadata,
                       nx, x, xi_guess, map_to_parent,
                       h0, d_val, R, y_cap,
                       reg_eps, debug_monitor, step_idx):
    """
    Solve ray-tracing projection with given parameters.
    """
    dxi = ufl.TestFunction(V_xi)
    
    # Capped parabola
    parabola_height = xi**2 / (2*R)
    capped_height = ufl.min_value(parabola_height, y_cap)
    Y_ind = ufl.as_vector([xi, (h0 - d_val) + capped_height])
    
    # Integration measure
    ds = ufl.Measure("ds", domain=parent_mesh, subdomain_data=facet_marker, metadata=metadata)
    ds_contact = ds(c_tag)
    
    # Tangent vector (perpendicular to current normal)
    # The ray-tracing condition (Y_ind - x) · tx = 0 finds ξ such that
    # the vector from current surface point x to indenter Y_ind(ξ) is 
    # parallel to the current normal nx (perpendicular to tangent tx)
    tx = ufl.as_vector([nx[1], -nx[0]])
    
    # Regularized residual
    # Ray-tracing condition: (Y_ind - x) · tx = 0 finds ξ such that projection is along normal
    ray_cond = ufl.dot(Y_ind - x, tx)
    
    # Regularization: penalize ξ far from current x-coordinate (biases toward vertical projection)
    x_scalar = x[0]
    reg_term = reg_eps * (xi - x_scalar)
    Res = (ray_cond + reg_term) * dxi * ds_contact
    
    entity_maps = [map_to_parent]
    
    # Solver options
    nonlin_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-9,
        "snes_max_it": 100,
        "snes_linesearch_max_it": 30,
        # "snes_linesearch_damping": 0.8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    if debug_monitor:
        nonlin_opts["snes_monitor"] = None
    
    problem = NonlinearProblem(
        F=Res,
        u=xi,
        bcs=[],
        petsc_options=nonlin_opts,
        petsc_options_prefix="ray-tracing_",
        entity_maps=entity_maps
    )
    
    # Initial guess 
    if xi_guess is None:
        xi.interpolate(lambda coords: coords[0])
    else:
        xi.x.array[:] = xi_guess.x.array[:]
    xi.x.scatter_forward()
    
    print(f"\n{'='*60}")
    print(f"Ray-tracing projection at step {step_idx}, d = {d_val:.4f}")
    print(f"  Initial xi: min={xi.x.array.min():.4f}, max={xi.x.array.max():.4f}")
    
    try:
        problem.solve()
        reason = problem.solver.getConvergedReason()
        num_its = problem.solver.getIterationNumber()
        
        print(f"  Convergence reason: {reason}")
        print(f"  Number of iterations: {num_its}")
        print(f"  Final xi: min={xi.x.array.min():.4f}, max={xi.x.array.max():.4f}")


        # CRITICAL: Detect collapsed solutions before returning
        xi_range = xi.x.array.max() - xi.x.array.min()
        xi_expected_range = X_dof.max() - X_dof.min()
        
        if xi_range < 0.1 * xi_expected_range:
            # Solution collapsed to trivial state - reject immediately
            print(f"  CRITICAL: xi collapsed to zero/trivial solution (range={xi_range:.4f}, expected ~{xi_expected_range:.4f})")
            print(f"{'='*60}")
            # Restore initial guess
            if xi_guess is not None:
                xi.x.array[:] = xi_guess.x.array[:]
            else:
                xi.x.array[:] = X_dof 
            xi.x.scatter_forward()
            return False, -99  # Special code for collapsed solution
        
        converged = reason > 0
        print(f"{'='*60}")
        return converged, reason
        
    except Exception as e:
        print(f"\n  ERROR in ray-tracing solve: {e}")
        print(f"{'='*60}")
        return False, -100

def _plot_geometry_debug_sliding(xi_old, xi_new, h0, d_val, R, y_cap, 
                                  step_idx, output_dir, ndim, u_field=None):
    """
    Plot ray-tracing geometry with capped shifted indenter.
    
    Now includes current (deformed) configuration plots in addition to reference configuration.
    Layout: 3 rows x 2 columns
      - Row 1: Reference configuration (full domain)
      - Row 2: Reference configuration (contact zone)
      - Row 3: Current (deformed) configuration (contact zone)
    """
    
    V_xi = xi_new.function_space
    dof_coords = V_xi.tabulate_dof_coordinates()
    x_ref = dof_coords[:, 0]
    y_ref = dof_coords[:, 1]
    
    xi_old_vals = xi_old.x.array
    xi_new_vals = xi_new.x.array
    
    # Indenter profiles
    parabola_old = (x_ref)**2 / (2*R)
    parabola_new = (xi_new_vals)**2 / (2*R)
    y_ind_old = h0 + np.minimum(parabola_old, y_cap)
    y_ind_new = (h0 - d_val) + np.minimum(parabola_new, y_cap)
    
    xi_cutoff = np.sqrt(2*R*y_cap)
    contact_zone_margin = 2.0 * xi_cutoff
    
    # === EXTRACT CURRENT CONFIGURATION ===
    # Evaluate displacement on contact mesh if available
    x_curr = x_ref.copy()
    y_curr = y_ref.copy()
    
    if u_field is not None:
        try:
            u_at_dofs = u_field.x.array.reshape(-1, ndim)
            x_curr = x_ref + u_at_dofs[:, 0]
            y_curr = y_ref + u_at_dofs[:, 1]
            # print(f"  [Plot] Max |u_x|={np.abs(u_at_dofs[:,0]).max():.4e}, max |u_y|={np.abs(u_at_dofs[:,1]).max():.4e}")

        except Exception as e:
            print(f"  [Plot] Warning: Could not evaluate displacement at any points")
            
    
    # === CREATE FIGURE ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    fig.suptitle(f"Ray-Tracing Geometry (step {step_idx}, d={d_val:.3f})", fontsize=16)
    
    sort_idx = np.argsort(x_ref)
    
    # Zooming 
    zoom_margin = 3.0 * xi_cutoff
    x_zoom_min = -zoom_margin
    x_zoom_max = +zoom_margin
    
    in_zoom = (x_ref >= x_zoom_min) & (x_ref <= x_zoom_max)
    x_zoom = x_ref[in_zoom]
    xi_old_zoom = xi_old_vals[in_zoom]
    xi_new_zoom = xi_new_vals[in_zoom]
    sort_zoom = np.argsort(x_zoom)
    
    # REFERENCE configuration -- indenter, contact surface and projected points
    ax1 = axes[0]
    ax1.set_title("REFERENCE CONFIG")
    
    y_zoom = y_ref[in_zoom]
    y_ind_zoom = y_ind_new[in_zoom]
    y_ind_old_zoom = y_ind_old[in_zoom]

    ax1.plot(x_zoom[sort_zoom], y_zoom[sort_zoom], 'b-o', 
             label='Contact surface (ref)', markersize=4)
    
    xi_dense_zoom = np.linspace(x_zoom_min, x_zoom_max, 200)
    parabola_zoom = (xi_dense_zoom)**2 / (2*R)
    # y_ind_dense_zoom = (h0 - d_val) + np.minimum(parabola_zoom, y_cap)
    y_ind_dense_zoom = (h0) + np.minimum(parabola_zoom, y_cap)
    ax1.plot(xi_dense_zoom, y_ind_dense_zoom, 'k-', linewidth=2, label='Indenter')
    
    # ax1.plot(xi_new_zoom, y_ind_zoom, 'g^', markersize=6, label=r'$Y(\xi)$')
    ax1.plot(x_zoom, y_ind_old_zoom, 'g^', markersize=6, label=r'$Y(\xi)$')
    
    step = max(1, len(x_zoom) // 20)
    for i in range(0, len(x_zoom), step):
        # ax1.plot([x_zoom[i], xi_new_zoom[i]], [y_zoom[i], y_ind_zoom[i]], 
        #          'g--', linewidth=1, alpha=0.6)
        ax1.plot([x_zoom[i], x_zoom[i]], [y_zoom[i], y_ind_old_zoom[i]], 
                 'g--', linewidth=1, alpha=0.6)
    
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    # ax1.set_xlim(x_zoom_min, x_zoom_max)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.legend(loc='best', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # CURRENT configuration -- indenter, contact surface and projected points
    # Zoom mask for current configuration
    in_zoom_curr = (x_ref >= x_zoom_min) & (x_ref <= x_zoom_max)
    x_curr_zoom = x_curr[in_zoom_curr]
    y_curr_zoom = y_curr[in_zoom_curr]
    x_ref_zoom_curr = x_ref[in_zoom_curr]
    xi_new_zoom_curr = xi_new_vals[in_zoom_curr]
    sort_curr = np.argsort(x_curr_zoom)
    

    ax2 = axes[1]
    ax2.set_title("CURRENT CONFIG")
    
    # Indenter profile at projection points
    y_ind_proj = (h0 - d_val) + np.minimum((xi_new_zoom_curr)**2 / (2*R), y_cap)
    
    ax2.plot(x_curr_zoom[sort_curr], y_curr_zoom[sort_curr], 'b-o', 
                label='Contact surface (cur)', markersize=4)
    ax2.plot(xi_dense_zoom, y_ind_dense_zoom - d_val, 'k-', linewidth=2, label='Indenter')
    ax2.plot(xi_new_zoom_curr[sort_curr], y_ind_proj[sort_curr], 'g^', 
                markersize=6, label=r'$y(\xi)$')
    
    # Draw projection lines
    step = max(1, len(x_curr_zoom) // 20)
    for i in range(0, len(x_curr_zoom), step):
        idx = sort_curr[i]
        ax2.plot([x_curr_zoom[idx], xi_new_zoom_curr[idx]], 
                    [y_curr_zoom[idx], y_ind_proj[idx]], 
                    'g--', linewidth=1, alpha=0.6)
    
    ax2.set_xlabel(r'$x$ (current)')
    ax2.set_ylabel(r'$y$ (current)')
    # ax2.set_xlim(x_zoom_min, x_zoom_max)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rt_geometry_step{step_idx:04d}.png", dpi=150)
    plt.close()



