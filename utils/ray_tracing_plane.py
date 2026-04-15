#!/usr/bin/env python3
"""
Ray-tracing subproblem for contact detection
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from dolfinx import fem
import ufl
import basix
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc


def ray_tracing_mapping(
    parent_mesh, 
    contact_mesh, 
    facet_marker, 
    c_tag, 
    metadata, 
    nx,              # UFL normal expression on parent mesh
    x,               # UFL current position (X + u) on parent mesh
    xi_guess,        # Previous solution as initial guess (fem.Function or None)
    map_to_parent,   # Entity map for submesh assembly
    d_val,           # Current indentation depth
    plane_loc,       # Plane location (y = -plane_loc)
    R,               # Half-Disk radius
    step_idx,        # Current step index for filenames
    output_dir,      # Output directory for plots
    debug_plots=True,
    debug_monitor=False,
    u_field=None,
    # === SOLVER PARAMETERS ===
    reg_eps=0.0,        # Regularization strength (increased default)
):
    """
    Ray-tracing projection strategy.
    Nonlinear equation solved with Newton's method.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ndim = parent_mesh.geometry.dim
    element_deg = 1  # Linear elements for convective coordinate 

    # Scalar function space on submesh
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
    
    # === TRY SOLVING WITH INCREASING RELAXATION ===
    converged = False
    final_reason = -1

    converged, reason = _solve_ray_tracing(
        xi, xi_old, V_xi, X_dof,
        parent_mesh, contact_mesh, facet_marker, c_tag, metadata,
        nx, x, xi_guess, map_to_parent,
        d_val, plane_loc, R,
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
            _plot_geometry_debug(
                parent_mesh, contact_mesh, facet_marker, c_tag,
                xi_old, xi, d_val, plane_loc, R, step_idx, output_dir, ndim,
                nx = nx,
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
                       d_val, plane_loc, R,
                       reg_eps, debug_monitor, step_idx):
    """
    Solve ray-tracing projection with given parameters.
    """
    dxi = ufl.TestFunction(V_xi)
    
    # Horizontal plane at y=-plane_loc
    Y_plane = ufl.as_vector([xi, plane_loc * np.ones_like(xi)])
    
    # Integration measure
    ds = ufl.Measure("ds", domain=parent_mesh, subdomain_data=facet_marker, metadata=metadata)
    ds_contact = ds(c_tag)
    
    # Tangent vector
    tx = ufl.as_vector([nx[1], -nx[0]])
    
    # Regularized residual
    ray_cond = ufl.dot(Y_plane - x, tx)
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
    print(f"  Solving with standard Newton solver")
    print(f"  Regularization eps={reg_eps:.2e}")
    print(f"{'='*60}")
    
    try:
        problem.solve()
        reason = problem.solver.getConvergedReason()
        num_its = problem.solver.getIterationNumber()
        
        print("--- Monitoring ray-tracing mapping ---")
        print(f"  Convergence reason: {reason}, number of iterations: {num_its}")
        print(f"  Final xi: min={xi.x.array.min():.4f}, max={xi.x.array.max():.4f}")
        
        # CRITICAL: Detect collapsed solutions before returning
        xi_range = xi.x.array.max() - xi.x.array.min()
        xi_expected_range = X_dof.max() - X_dof.min()
        
        if xi_range < 0.1 * xi_expected_range:
            # Solution collapsed to trivial state - reject immediately
            print(f"  CRITICAL: xi collapsed to zero/trivial solution (range={xi_range:.4f}, expected ~{xi_expected_range:.4f})")
            # Restore initial guess
            if xi_guess is not None:
                xi.x.array[:] = xi_guess.x.array[:]
            else:
                xi.x.array[:] = X_dof
            xi.x.scatter_forward()
            return False, -99  # Special code for collapsed solution
        
        converged = reason > 0
        return converged, reason
        
    except Exception as e:
        print(f"\n  ERROR in ray-tracing solve: {e}")
        return False, -100

def _plot_geometry_debug(parent_mesh, contact_mesh, facet_marker, c_tag,
                         xi_old, xi_new, d_val, plane_loc, R,
                         step_idx, output_dir, ndim, nx, u_field=None):
    
    
    V_xi = xi_new.function_space
    dof_coords = V_xi.tabulate_dof_coordinates()
    x_ref = dof_coords[:, 0]
    y_ref = dof_coords[:, 1]
    
    xi_old_vals = xi_old.x.array
    xi_new_vals = xi_new.x.array
    
    sort_idx = np.argsort(x_ref)

    x_disk_coords = np.linspace(-R, R, 200)
    y_disk_coords = -np.sqrt(R**2 - x_disk_coords**2) + R

    # === EXTRACT CURRENT CONFIGURATION ===
    # Evaluate displacement on contact mesh if available
    x_curr = x_ref.copy()
    y_curr = y_ref.copy()
    has_current_config = False

    if u_field is not None:
        try:
            u_at_dofs = u_field.x.array.reshape(-1, ndim)
            x_curr = x_ref + u_at_dofs[:, 0]
            y_curr = y_ref + u_at_dofs[:, 1]
            print(f"  [Plot] Max |u_x|={np.abs(u_at_dofs[:,0]).max():.4e}, max |u_y|={np.abs(u_at_dofs[:,1]).max():.4e}")
            has_current_config = True

        except Exception as e:
            print(f"  [Plot] Warning: Could not evaluate displacement at any points")
            has_current_config = False

    # =========================================================================
    # FIGURE 1: CURRENT CONFIGURATION (ZOOMED TO CONTACT AREA)
    # =========================================================================
    # # Increase font sizes for better readability
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.labelsize'] = 26
    # plt.rcParams['axes.titlesize'] = 16
    # plt.rcParams['legend.fontsize'] = 14
    # plt.rcParams['xtick.labelsize'] = 11
    # plt.rcParams['ytick.labelsize'] = 11
    # plt.rcParams['mathtext.fontset'] = 'cm'       # Computer Modern -- matches LaTeX appearance
    # plt.rcParams['font.family'] = 'serif'
    if has_current_config:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        # fig1.suptitle(f"Ray-Tracing: CURRENT Configuration (step {step_idx}, d={d_val:.4f})")
        
        sort_idx_c = np.argsort(x_curr)
        ax1.plot(x_curr[sort_idx_c], y_curr[sort_idx_c], '-ro', 
                 label=r"slave boundary $\mathbf{x}^1$", markersize=6, alpha=0.6)
        # ax1.plot(x_disk_coords, y_disk_coords - d_val, 'k--', linewidth=0.5, alpha=0.5)#'b--', linewidth=2, label='Indenter underside (half-disk)')
        # ax1.plot(x_ref[sort_idx], y_plane, 'k-', linewidth=2.0)
        ax1.plot(np.linspace(-1, 1, 100), plane_loc * np.linspace(-1, 1, 100), 'k-', linewidth=1.0, alpha=0.7)
        ax1.set_title(f"Ray-tracing mapping (current configuration)", fontsize=16)
        # Plot projection rays in reference configuration (sampled to avoid clutter)
        try:
            n_samples = min(50, len(x_curr))  # Increased from 20 to 50
            step = max(1, len(x_curr) // n_samples)
            mask_R_val = 0.8 * R 
            for i in range(0, len(x_curr), step):
                x0 = x_curr[i]
                # only plot rays for reference x within [-R, R]
                if x0 < -mask_R_val or x0 > mask_R_val:
                    continue
                y0 = y_curr[i]
                xi_proj = xi_new_vals[i]
                y_proj = plane_loc
                ax1.plot([x0, xi_proj], [y0, y_proj], color='gray', linestyle='--', linewidth=1.0)
        except Exception:
            pass
        ax1.plot(xi_new_vals, plane_loc * np.ones_like(xi_new_vals), 'x', color='green', markersize=6, label=r"mapped master points $\mathbf{x}^2(\xi^h)$")
        # ax1.plot(x_ref[sort_idx], y_plane, 'o', color='black', markersize=4,
        #      markerfacecolor='None', label=r'$x^2(\xi)$ (initial)')
        ax1.set_xlabel(r'$x$', fontsize=16)
        ax1.set_ylabel(r'$y$', fontsize=16)
        ax1.set_xlim(-1, 1)  # Zoom to contact area
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.legend(
            loc='upper right',
            frameon=True,
            framealpha=0.95,
            edgecolor='0.7',
            facecolor='white',
            borderpad=0.35,
            labelspacing=0.3,
            handlelength=1.8,
            handletextpad=0.6
        )
        # ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rt_geometry_current_step{step_idx:04d}.png", dpi=150)
        plt.close()

        # =========================================================================
        # FIGURE 2: CONTACT PARAMETER IN CURRENT CONFIGURATION
        # =========================================================================
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        # fig2.suptitle(f"Ray-Tracing: Contact Parameter (step {step_idx}, d={d_val:.4f})")
        ax2.set_title(r"Evolution of convective coordinate $\xi$")
        
        print(f" [Plot] xi_old_vals: min={xi_old_vals.min():.6f}, max={xi_old_vals.max():.6f}")
        print(f" [Plot] xi_new_vals: min={xi_new_vals.min():.6f}, max={xi_new_vals.max():.6f}")
        
        ax2.plot(x_ref[sort_idx], xi_old_vals[sort_idx], '--o', color='black', 
                label=r'$\xi$ (initial)', markersize=4, alpha=0.7)
        ax2.plot(x_ref[sort_idx], xi_new_vals[sort_idx], 'x', color='green', 
                label=r'$\xi$ (solved)', markersize=4)
        # ax2.plot(x_ref[sort_idx], x_ref[sort_idx], 'k:', 
        #          linewidth=2, label=r'$\xi = X (ideal)')
        # ax2.axvline(x=0.0, color='orange', linestyle='--', alpha=0.5, label='Center (x=0)')
        ax2.set_xlabel(r'Reference $X$')
        ax2.set_ylabel(r'$\xi$')
        # ax2.set_xlim(-0.75, 0.75)  # Zoom to contact area
        ax2.legend(fontsize='small')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rt_geometry_contact_param_step{step_idx:04d}.png", dpi=150)
        plt.close()

        # ------------------------------------------------------------------------ #
