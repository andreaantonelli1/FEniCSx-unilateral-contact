#!/usr/bin/env python3
"""
Signorini Contact: Compressible Hyperelastic Disk on Rigid Plane
======================================================================

This script implements a compressible hyperelastic contact simulation using:
- Standard displacement formulation (no pressure multiplier)
- Half-disk mesh (from mesh.py)
- Configurable contact method (Nitsche or Augmented Lagrangian), with RAY-TRACING mapping
- Strain energy from constitutive.py 

Structure:
1. Configuration (SimulationConfig)
2. Contact problem (ContactProblem)
3. Solver (ContactSolver)
4. Main simulation loop (run_simulation)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import argparse
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import basix
import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh as dolfinx_mesh
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import Constant, Function, Expression, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags

from utils import (
    half_disk_mesh,
    convert_mesh_new,
    compute_kinematics,
    get_constitutive_model,
    compute_stress,
    compute_stress_linear,
    pneg,
    create_contact_submesh,
    gap_rt,
    extract_boundary_cells,
    expression_at_quadrature,
    project_vector_on_boundary,
    ray_tracing_mapping,
)

# =============================================================================
# CONVERGENCE STATUS ENUM
# =============================================================================

class ConvergenceStatus(Enum):
    """Status codes for solve_step."""
    CONVERGED = 0
    MAX_ITERATIONS = 1
    STAGNATED = 2
    SNES_FAILED = 3
    EXCEPTION = 4


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """All simulation parameters."""
    
    # Loading
    d_final: float = 0.2     # Final indentation depth
    n_steps: int = 20        # Number of load steps
    
    # Geometry - Half-disk
    sym: bool = False           # Whether to use symmetric half-disk mesh
    R: float = 1.0              # Disk radius
    plane_loc: float = 0.0      # Rigid plane location (y-coordinate)
    LcMin: float = 0.01         # Min mesh size (contact region)
    LcMax: float = 0.2          # Max mesh size (far from contact)
    a: float = np.sqrt(2 * R * d_final)  # Estimated contact half-width
    Dmin: float = a             # Refinement distance min
    Dmax: float = 1.1 * Dmin    # Refinement distance max
    
    # Material
    linear: bool = False               # Whether to use linear elastic model (small strain)
    hyperelastic_model: str = "NH_ss"  # check constitutive.py for available models 
    Y: float = 1.0                     # Young's modulus 
    nu: float = 0.3                    # Poisson's ratio
    
    # FE degree
    u_degree: int = 1
    lN_degree: int = 1

    # Quadrature degree
    quad_degree_c: int = 4  # Contact integrals (ds)
    quad_degree_v: int = 4  # Volume integrals  (dx)
    
    # Contact method selection
    contact_method: str = "nitsche"
    
    # Nitsche contact parameters
    theta: int = 0
    gamma0: float = 10.0        # gamma = gamma0 * Y / h
    
    # Augmented Lagrangian (with Uzawa) contact parameters
    al_coeff: float = 1.0    # Adimensional coefficient for scaling
    uzawa_penalty: float = al_coeff * Y / LcMin
    aug_lag_penalty: float = al_coeff * Y 
    aug_lag_max_iter: int = 50
    aug_lag_tol_gap: float = 1e-2 * LcMin # % of the minimum mesh size
    aug_lag_tol_lN: float = 1e-2          # Tolerance for Lagrange multiplier convergence (relative change)
    
    # Solver
    max_iter: int = 50
    snes_rtol: float = 1e-9
    snes_atol: float = 1e-9
    use_alternating: bool = True  # Use alternating solution scheme 
    
    # Output
    output_dir: str = "output_signorini_" + contact_method
    debug_plots: bool = True    # Ray-tracing debug plots (in output_dir/ray_tracing)
        
    def recompute_derived(self):
        """Recompute parameters derived from base settings (call after CLI overrides)."""
        self.uzawa_penalty = self.al_coeff * self.Y / self.LcMin
        self.aug_lag_tol_gap = 1e-2 * self.LcMin

# =============================================================================
# CONTACT PROBLEM DEFINITION
# =============================================================================

class ContactProblem:
    """
    Contact of half-disk on rigid plane.

    Compressible formulation (displacement only, no pressure multiplier).
    """
    
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.comm = MPI.COMM_WORLD
        
        self._setup_mesh()
        self._setup_spaces()
        self._setup_kinematics()
        self._setup_contact_geometry()

        self.ca_quad_prev = None  # For tracking contact activity changes in SNES monitor
    
    def _setup_mesh(self):
        """Generate half-disk mesh using gmsh."""
        cfg = self.cfg
        
        # Create mesh file in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_file = Path(tmpdir) / "half_disk.msh"
            
            half_disk_mesh(
                R=cfg.R,
                LcMin=cfg.LcMin,
                LcMax=cfg.LcMax,
                Dmin=cfg.Dmin,
                Dmax=cfg.Dmax,
                filename=mesh_file
            )
            
            # Convert to XDMF
            xdmf_file = mesh_file.with_suffix(".xdmf")
            convert_mesh_new(mesh_file, xdmf_file, gdim=2)
            
            # Read mesh
            with XDMFFile(self.comm, xdmf_file, "r") as xdmf:
                self.domain = xdmf.read_mesh()
        
        # Define boundary markers for half-disk full mesh
        self.ft = {"top": 1, "contact": 2}
    
    
        def top_boundary(x):
            return np.isclose(x[1], cfg.R)
        
        def contact_boundary(x):   
            return x[1] < 0.8 * cfg.R   
        
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        
        top_facets = locate_entities_boundary(self.domain, fdim, top_boundary)
        contact_facets = locate_entities_boundary(self.domain, fdim, contact_boundary)
        
        top_values = np.full(len(top_facets), self.ft["top"], dtype=np.int32)
        contact_values = np.full(len(contact_facets), self.ft["contact"], dtype=np.int32)
        
        indices = np.concatenate([top_facets, contact_facets])
        values = np.hstack([top_values, contact_values])
        sorted_idx = np.argsort(indices)
        
        self.facet_tags = meshtags(self.domain, fdim, indices[sorted_idx], values[sorted_idx])
                
        if self.comm.rank == 0:
            n_cells = self.domain.topology.index_map(tdim).size_global
            print(f"Half-disk mesh: {n_cells} cells, h_min={cfg.LcMin}")
            print(f"Displacement formulation: P{cfg.u_degree} (compressible)")
            
            # Save mesh for inspection
            os.makedirs(cfg.output_dir, exist_ok=True)
            mesh_path = os.path.join(cfg.output_dir, "mesh.xdmf")
            with XDMFFile(self.comm, mesh_path, "w") as xdmf:
                xdmf.write_mesh(self.domain)
                xdmf.write_meshtags(self.facet_tags, self.domain.geometry)
            print(f"Mesh saved to {mesh_path}")
    
    def _setup_spaces(self):
        """Create function spaces and measures."""
        cfg = self.cfg
        ndim = self.domain.topology.dim

        # Create Contact submesh
        self.contact_mesh, self.map_to_parent, self.V_sub, self.V_parent, self.interp_data, self.n_sub_cells = \
            create_contact_submesh(self.domain, self.facet_tags, self.ft["contact"], 1, ndim)
         
        # Create displacement and LM function spaces
        self.V_u = functionspace(self.domain, ("CG", cfg.u_degree, (2,)))      
        self.V_lm = functionspace(self.contact_mesh, ("CG", cfg.lN_degree)) 
        # Combining them into a mixed space
        self.W = ufl.MixedFunctionSpace(self.V_u, self.V_lm)

        # Create functions for displacement and Lagrange multiplier
        self.u = Function(self.V_u, name="u")
        self.u_old = Function(self.V_u, name="u_old")
        self.lN = Function(self.V_lm, name="LM")
        if cfg.contact_method == "augmented_lagrangian":
            # Test functions
            self.du, self.dlN = ufl.TestFunctions(self.W)
            # Trial functions
            self.delta_u, self.delta_lN = ufl.TrialFunctions(self.W)
        else:
            # Test function
            self.du = ufl.TestFunction(self.V_u) 

        # Create stress function space for post-processing (DG of degree u_degree-1, tensor-valued)
        self.V_stress = functionspace(self.domain, ("DG", (cfg.u_degree - 1), (ndim, ndim)))  
        self.sigma_DG = Function(self.V_stress)

        # Measures on parent mesh with appropriate quadrature degree  
        self.metadata = {"quadrature_degree": cfg.quad_degree_c}
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags, metadata=self.metadata)
        self.dx = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": cfg.quad_degree_v}) 

        # Other Contact spaces for post-processing and interpolation
        # DG0 for normals
        self.V_sub_n = functionspace(self.contact_mesh, ("DG", 0, (2,))) # DG0 - vector space on submesh
        self.V_parent_n = functionspace(self.domain, ("DG", 0, (2,)))    # DG0 - vector space on parent
        sub_cell_map = self.contact_mesh.topology.index_map(self.contact_mesh.topology.dim)
        self.num_sub_cells = sub_cell_map.size_local + sub_cell_map.num_ghosts
        # For parent -> submesh interpolation
        self.interp_data_n = fem.create_interpolation_data(
            self.V_sub_n, self.V_parent_n, cells=np.arange(self.num_sub_cells)
        )
        # Quadrature space on contact boundary for LM post-processing
        q_element = basix.ufl.quadrature_element(
            self.contact_mesh.basix_cell(), value_shape=(), scheme="default", degree=cfg.quad_degree_c
        )
        self.V_q = functionspace(self.contact_mesh, q_element)

        # Dirichlet BC: fix top boundary (will be updated with displacement)
        fdim = ndim - 1
        top_facets = self.facet_tags.find(self.ft["top"])
        
        self.top_dofs = fem.locate_dofs_topological(self.V_u, fdim, top_facets)
        self.disp_bc = Function(self.V_u)
        self.disp_bc.x.array[:] = 0.0
        self.bcs = [fem.dirichletbc(self.disp_bc, self.top_dofs)]
    
    def _setup_kinematics(self):
        """Setup large deformation kinematics."""
        cfg = self.cfg
        
        # Get kinematics from constitutive.py
        kin = compute_kinematics(self.u, self.domain.geometry.dim, cfg.linear)
        self.F = kin['F']  # Deformation gradient
        self.J = kin['J']  # Jacobian of the trasformation J = det(F)
        self.C = kin['C']  # Right Cauchy-Green deformation tensor
        self.E = kin['E']  # Green-Lagrange strain tensor
        
        # Get strain energy from constitutive.py (compressible)
        if cfg.linear:
            cfg.hyperelastic_model = "LINEAR"  # Override to use linear elastic model in constitutive.py
        
        self.psi, self.psi_dev, self.psi_vol, self.mat_params = get_constitutive_model(
            cfg.hyperelastic_model, self.domain, kin, cfg.Y, cfg.nu
        )
        
        # Stress measures
        if cfg.linear:
            self.P, self.S, self.sigma = compute_stress_linear(self.psi, self.mat_params['eps']) # P == S == sigma for linear elasticity
        
        else:
            self.P, self.S, self.sigma = compute_stress(self.psi, self.F, self.J)
        
        # Residual: momentum balance only 
        self.R_elastic = ufl.inner(self.P, ufl.grad(self.du)) * self.dx
    
    def _setup_contact_geometry(self):
        """Setup contact geometry and residual."""
        cfg = self.cfg
        
        # Spatial coordinates in reference configuration
        self.X = ufl.SpatialCoordinate(self.domain)
        
        # Out-of-plane normal
        self.Ny = ufl.as_vector([0, 1])
        # Plane position vector (fixed, rigid plane)
        self.y = ufl.as_vector([self.X[0], cfg.plane_loc * np.ones_like(self.X[0])])

        # Half-disk reference normal
        self.Nx = ufl.FacetNormal(self.domain)
        
        # Current normal
        self.cofactor = self.J * ufl.inv(self.F).T
        nx_ = ufl.dot(self.cofactor, self.Nx)
        self.nx = nx_ / ufl.sqrt(ufl.dot(nx_, nx_))

        if cfg.linear:
            # For linear elasticity, current normal == reference normal 
            self.nx = self.Nx
    

        # Normal traction (from 1st Piola-Kirchhoff stress)
        self.P_N = ufl.dot(self.P, self.Nx)
        # Normal component of traction (contact pressure)
        self.P_n = ufl.dot(self.P_N, self.nx) 
        
        # Characteristic mesh size
        self.h = ufl.CellDiameter(self.domain)
        
        # Setup contact residual (only called for Nitsche or Augmented Lagrangian)
        if cfg.contact_method == "snes_vi":
            self.R_total = self.R_elastic
        else:    
            self._setup_contact_residual()
    
    def _setup_contact_residual(self):
        """Build contact residual based on selected method."""
        cfg = self.cfg
        ds_contact = self.ds(self.ft["contact"]) # Measure on contact boundary 
        
        # Function to hold current y position on contact boundary (for gap evaluation)
        self.y_func = Function(self.V_sub) 
        # Gap function for ray-tracing mapping
        g = gap_rt(self.nx, self.X + self.u, self.y_func)
        
        if cfg.contact_method == "nitsche":  
            # Nitsche penalty parameter
            self.gamma = cfg.gamma0 * cfg.Y / self.h
            # self.gamma = cfg.gamma0 * cfg.Y / cfg.LcMin  # positive constant, not mesh-dependent 
            
            # Nitsche contact formulation
            P_n_variation = ufl.derivative(self.P_n, self.u, self.du)
            term1 = - (cfg.theta * self.P_n * P_n_variation / self.gamma) * ds_contact
            
            self.contact_arg = self.P_n + self.gamma * g
            contact_arg_variation = ufl.derivative(
                cfg.theta * self.P_n + self.gamma * g,
                self.u,
                self.du
            )
            term2 = (pneg(self.contact_arg) * contact_arg_variation / self.gamma) * ds_contact
            
            self.R_contact = term1 + term2
            self.R_total = self.R_elastic + self.R_contact
            
        elif cfg.contact_method == "uzawa":
            # Augmented Lagrangian with Uzawa's augmentation loop contact formulation
            self.V_sub_scalar = functionspace(self.contact_mesh, ("CG", 1))
            self.lN_sub = Function(self.V_sub_scalar, name="lN_sub") # Lagrange multiplier for contact pressure (defined on submesh)
            
            self.eps_contact = cfg.uzawa_penalty # Augmentation parameter 
            # self.contact_arg = self.lN_contact + self.eps_contact * g
            self.contact_arg = self.lN_sub + self.eps_contact * g
            g_variation = ufl.derivative(g, self.u, self.du)

            self.R_contact = pneg(self.contact_arg) * g_variation * ds_contact
            self.R_total = self.R_elastic + self.R_contact

        elif cfg.contact_method == "augmented_lagrangian":

            self.eps_contact = cfg.aug_lag_penalty # Augmentation parameter 
            
            self.contact_arg = self.lN + self.eps_contact * g
            # g_variation = ufl.dot(self.nx, self.du)  
            g_variation = ufl.derivative(g, self.u, self.du)  # More general variation of the gap function

            self.R_elastic += pneg(self.contact_arg) * g_variation * ds_contact
            self.R_elastic += (pneg(self.contact_arg) - self.lN) * self.dlN / self.eps_contact * ds_contact 
            self.residual = ufl.extract_blocks(self.R_elastic)
            # define the jacobian
            jac = ufl.derivative(self.R_elastic, self.u, self.delta_u) + ufl.derivative(self.R_elastic, self.lN, self.delta_lN)
            self.Jac = ufl.extract_blocks(jac)  
        else:
            raise ValueError(f"Unknown contact method: {cfg.contact_method}")
            
    def project_current_normal_to_function(self):
        """
        Project the current (deformed) normal onto a DG function space.
        This allows it to be used in Expression objects.
        """
        
        # Project the current normal (self.nx) onto DG space
        nx_func = project_vector_on_boundary(
            vector_expr=self.nx,  # Your current normal expression
            V=self.V_parent_n,  # DG function space on parent mesh
            facet_tags=self.facet_tags,
            facet_id=self.ft["contact"],
            metadata=self.metadata
        )
        
        return nx_func

    def store_state(self):
        """Store current state for next step."""
        self.u_old.x.array[:] = self.u.x.array[:]
        self.u_old.x.scatter_forward()


# =============================================================================
# SOLVER
# =============================================================================

class ContactSolver:
    """Newton solver for contact problem."""
    
    def __init__(self, problem: ContactProblem, cfg: SimulationConfig):
        self.problem = problem
        self.cfg = cfg
        self.comm = problem.comm
        self.xi = None
        # Tracking for augmented-lagrangian metrics (per load step)
        self.last_aug_iters = 0
        self.total_newton_its = 0.0
        
        if cfg.contact_method == "snes_vi":
            self.petsc_options = {
            "snes_type": "vinewtonrsls",  # Variational inequality solver (Newton with line-search on reduced space/active set)
            "snes_monitor": None,
            # VI-specific monitor
            # "snes_vi_monitor": None,  # Monitor for VI convergence (e.g. gap, residual norms)
            "snes_max_it": cfg.max_iter,
            "snes_rtol": cfg.snes_rtol,
            "snes_atol": cfg.snes_atol,
            # KSP (linear solver) options 
            "ksp_type": "preonly",                # Direct solver
            "pc_type": "lu",                      # LU factorization
            "pc_factor_mat_solver_type": "mumps", # Use MUMPS
        }
        else:
            self.petsc_options = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt", # backtracking line search
                "snes_monitor": None,
                "snes_max_it": cfg.max_iter,
                "snes_rtol": cfg.snes_rtol,
                "snes_atol": cfg.snes_atol,
                # KSP (linear solver) options
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                #### DEBUG OPTIONS ####
                ## debugging linear solver convergence
                # "ksp_monitor": None,  
                # "ksp_converged_reason": None,  
                ## debugging line search
                # "snes_linesearch_monitor": None, 
            }
    
    def solve_step(self, d_val, step):
        """
        Solve one load step with selected contact method. 
        
        Parameters
        ----------
        d_val : float
            Current indentation depth (applied as displacement BC)
        step : int
            Current load step index 
            
        Returns
        -------
        status : ConvergenceStatus
        n_iters : int
            Number of Newton iterations at convergence for the current step
        """
        cfg = self.cfg
        prob = self.problem
        
        # Update Dirichlet BC for indentation
        def disp_expr(x):
            return np.stack([np.zeros_like(x[0]), -d_val * np.ones_like(x[0])])
        
        prob.disp_bc.interpolate(disp_expr)
        prob.disp_bc.x.scatter_forward()
        
        # Solve
        if cfg.contact_method == "nitsche":
            return self._solve_standard(d_val, step)
        elif cfg.contact_method == "uzawa":
            return self._solve_uzawa(d_val, step)
        elif cfg.contact_method == "augmented_lagrangian":
            return self._solve_augmented_lagrangian(d_val, step) 
        else:  
            return self._solve_snes_vi()
    
    def _solve_standard(self, d_val, step):
        """Solve contact problem with alternating solution strategy
        
        Alternating strategy:
        1. Solve for contact mapping (ray-tracing) with current displacement
        2. Solve for displacement with current (fixed) contact mapping 
        3. Check convergence: if both displacement and contact mapping have stabilized, stop. Otherwise, iterate.
        """
        cfg, prob = self.cfg, self.problem

        # Alternating parameters
        max_alt_iter = 10 if cfg.use_alternating else 1
        alt_tol = 0.01 if cfg.use_alternating else 10 
        
        print(f"\n{'='*70}")
        print(f"Starting alternating scheme for d={d_val:.4f}")
        print(f"{'='*70}")
        
        # Store previous displacement and xi for convergence check
        u_prev = prob.u.x.array.copy()
        xi_prev = self.xi.x.array.copy() if self.xi is not None else None
        
        total_newton_its = 0

        # Create NonlinearProblem once outside the loop.
        nlp = NonlinearProblem(
            F=prob.R_total,
            u=prob.u,
            bcs=prob.bcs,
            petsc_options=self.petsc_options,
            petsc_options_prefix="contact_sig",
            entity_maps=[prob.map_to_parent]
        )
        # nlp._snes.setMonitor(self.snes_debug_monitor)

        # Quantities on submesh
        self.X_sub = ufl.SpatialCoordinate(prob.contact_mesh)
        self.u_sub = Function(prob.V_sub)
        try:
            self.u_sub.interpolate_nonmatching(prob.u, cells=np.arange(prob.num_sub_cells),
                                            interpolation_data=prob.interp_data)
        except Exception as e:
            print(f"Warning: interpolate_nonmatching u_sub failed: {e}")
            print("Falling back to zero displacement on submesh for postprocessing.")
            pass

        for alt_iter in range(max_alt_iter):
            # ================================================================
            # Step 1: Update ray-tracing mapping (xi solver)
            # ================================================================

            self.xi = ray_tracing_mapping(
                parent_mesh=prob.domain,
                contact_mesh=prob.contact_mesh,
                facet_marker=prob.facet_tags,
                c_tag=prob.ft['contact'],
                metadata=prob.metadata,
                nx=prob.nx,
                x=prob.X + prob.u,
                xi_guess=self.xi if self.xi is not None else None,
                map_to_parent=prob.map_to_parent,
                d_val=d_val,
                plane_loc=cfg.plane_loc,
                R=cfg.R,
                step_idx=step,
                output_dir=os.path.join(cfg.output_dir, "ray_tracing"),
                debug_plots=cfg.debug_plots if alt_iter == 0  else False,  # Only debug first iteration
                debug_monitor=False,
                u_field=self.u_sub
            )

            # Update positions of projected points on reference plane
            y = ufl.as_vector([self.xi, cfg.plane_loc * np.ones_like(self.xi)])

            y_sub = fem.Function(prob.V_sub)
            y_sub.interpolate(Expression(y, prob.V_sub.element.interpolation_points))
            # Update y_func for gap evaluation in contact residual
            prob.y_func.x.array[:] = y_sub.x.array[:]
            prob.y_func.x.scatter_forward()

            # ================================================================
            # Step 2: Solve contact problem for the displacement field
            # ================================================================
            try:
                u_vec = prob.u.x.petsc_vec.copy()
                nlp.solver.solve(None, u_vec)
                reason = nlp.solver.getConvergedReason()
                n_its = nlp.solver.getIterationNumber()
                total_newton_its += n_its

                prob.u.x.petsc_vec.array[:] = u_vec.array
                prob.u.x.scatter_forward()
                u_vec.destroy()

                if reason < 0:
                    print(f"  Alt iter {alt_iter+1}: SNES failed with reason {reason}")
                    return ConvergenceStatus.SNES_FAILED, total_newton_its, reason

            except Exception as e:
                if self.comm.rank == 0:
                    print(f"  Alt iter {alt_iter+1}: SNES exception: {e}")
                return ConvergenceStatus.EXCEPTION, total_newton_its, None
            
            # ================================================================
            # Step 3: Check convergence
            # ================================================================
            # Convergence requires BOTH displacement AND contact mapping to stabilize
            
            # Displacement-based metrics
            du = prob.u.x.array - u_prev
            du_norm = np.linalg.norm(du)
            u_norm = np.linalg.norm(prob.u.x.array)
            rel_change_u = du_norm / (u_norm + 1e-10)  
            
            # Contact mapping (xi) metrics
            if xi_prev is not None:
                dxi = self.xi.x.array - xi_prev
                dxi_norm = np.linalg.norm(dxi)
                xi_norm = np.linalg.norm(self.xi.x.array)
                rel_change_xi = dxi_norm / (xi_norm + 1e-10)
            else:
                # First iteration: compare with X_dofs
                V_xi = functionspace(prob.contact_mesh, ("CG", 1))
                X_dofs = V_xi.tabulate_dof_coordinates()[:, 0]
                xi_prev = X_dofs
                dxi_norm = np.linalg.norm(self.xi.x.array - xi_prev)
                xi_norm = np.linalg.norm(self.xi.x.array)
                rel_change_xi = dxi_norm / (xi_norm + 1e-10)

            print(f"  Alt iter {alt_iter+1}: Newton its={n_its}, "
                  f"||du||={du_norm:.4e} (rel={rel_change_u:.4e}), "
                  f"||dξ||={dxi_norm:.4e} (rel={rel_change_xi:.4e})")

            # Check convergence: BOTH criteria must be satisfied
            converged_u = rel_change_u < alt_tol
            converged_xi = rel_change_xi < alt_tol and xi_prev is not None # skip first iteration for xi convergence check
            
            if converged_u and converged_xi:
                print(f"  -> Alternating minimization converged in {alt_iter+1} iterations")
                print(f"  -> Total Newton iterations: {total_newton_its}")
                
                return ConvergenceStatus.CONVERGED, total_newton_its, reason

            # Update previous state for next iteration
            u_prev[:] = prob.u.x.array
            xi_prev = self.xi.x.array.copy()

        # Maximum iterations reached
        print(f"  -> Alternating minimization reached max iterations ({max_alt_iter})")
        print(f"  -> Final: rel_u={rel_change_u:.4e}, rel_ξ={rel_change_xi:.4e} (tol={alt_tol:.4e})")
        return ConvergenceStatus.MAX_ITERATIONS, total_newton_its, None
    
    def _solve_uzawa(self, d_val, step):
        """Solve contact problem using Uzawa's method (submesh-based multiplier updates)."""
        
        cfg, prob = self.cfg, self.problem
        # Collect Newton iterations for each augmented-lagrangian iteration
        newton_its_list = []
        
        # Store previous multiplier for convergence check 
        self.lN_sub_prev_func = Function(prob.V_sub_scalar)
        self.lN_sub_prev_func.x.array[:] = prob.lN_sub.x.array[:]
        self.lN_sub_prev_func.x.scatter_forward()
        # Measure on contact submesh (created once, reused)
        dx_sub = ufl.Measure("dx", domain=prob.contact_mesh)

        for aug_iter in range(cfg.aug_lag_max_iter):

            status, n_its, reason = self._solve_standard(d_val, step)

            if status != ConvergenceStatus.CONVERGED:
                return status, aug_iter, reason
            
            # store per-aug-iteration Newton iterations
            try:
                newton_its_list.append(int(n_its))
            except Exception:
                # fallback: ignore non-integer iteration info
                pass

            # Check convergence of Augmented Lagrangian
            # Interpolate displacement onto submesh
            self.X_sub = ufl.SpatialCoordinate(prob.contact_mesh)
            self.u_sub = Function(prob.V_sub)
            try:
                self.u_sub.interpolate_nonmatching(prob.u, cells=np.arange(prob.n_sub_cells),
                                                   interpolation_data=prob.interp_data)
            except Exception as e:
                if self.comm.rank == 0:
                    print(f"Warning: interpolate_nonmatching u_sub failed: {e}; using fallback zeros")
                self.u_sub.x.array[:] = 0.0
                self.u_sub.x.scatter_forward()

            # Build y on submesh (rigid plane at plane_loc)
            y = ufl.as_vector([self.xi, cfg.plane_loc * np.ones_like(self.xi)])
            y_sub = Function(prob.V_sub)
            y_sub.interpolate(Expression(y, prob.V_sub.element.interpolation_points))

            # Project current normal to DG function on submesh    
            self.nx_sub = Function(prob.V_sub_n)
            try:
                nx_parent = prob.project_current_normal_to_function()
                self.nx_sub.interpolate_nonmatching(
                    nx_parent, cells=np.arange(prob.num_sub_cells),
                    interpolation_data=prob.interp_data_n)
            except Exception as e:
                print(f"Warning: interpolate_nonmatching nx_sub failed: {e}")
                break
            
            # Compute gap on submesh at nodes (for multiplier update)
            gN_sub = Function(prob.V_sub_scalar) # same space as lN_sub
            g_expr = gap_rt(self.nx_sub, self.X_sub + self.u_sub, y_sub)
            try:
                expr_g = Expression(g_expr, prob.V_sub_scalar.element.interpolation_points)
                gN_sub.interpolate(expr_g)
            except Exception as e:
                print(f"Warning: Expression(g_expr) compile failed: {e}; setting gN_sub zeros")
                gN_sub.x.array[:] = 0.0
                gN_sub.x.scatter_forward()  
 
            # Penetration on contact submesh: L2 norm and max penetration (at quadrature points)
            pen_L2, pen_max = self._compute_penetration(d_val)

            # Store lN_sub BEFORE the update (for computing the multiplier change)
            self.lN_sub_prev_func.x.array[:] = prob.lN_sub.x.array[:]
            self.lN_sub_prev_func.x.scatter_forward()

            # 1) Compute the classical multiplier candidate
            lN_candidate = pneg(prob.lN_sub.x.array + prob.eps_contact * gN_sub.x.array)

            # 2) Compute relative change in lagrange multiplier for convergence check
            lN_diff = Function(prob.V_sub_scalar)
            lN_diff.x.array[:] = lN_candidate - self.lN_sub_prev_func.x.array
            lN_diff.x.scatter_forward()
            lN_change_sq = self.comm.allreduce(
                fem.assemble_scalar(fem.form(lN_diff**2 * dx_sub)), MPI.SUM)
            lN_change = np.sqrt(lN_change_sq)
            # L2 norm of candidate multiplier
            lN_cand_func = Function(prob.V_sub_scalar)
            lN_cand_func.x.array[:] = lN_candidate
            lN_cand_func.x.scatter_forward()

            lN_norm_sq = self.comm.allreduce(
                fem.assemble_scalar(fem.form(self.lN_sub_prev_func**2 * dx_sub)), MPI.SUM)
            lN_norm = np.sqrt(lN_norm_sq) + 1e-12
            
            lN_rel = lN_change / lN_norm
                        
            print(f"  Aug.Lag iter {aug_iter+1}: max penetration= {pen_max:.4e}, "
                  f"lN rel. change = {lN_rel:.4e}")
            ## Convergence check
            if pen_max < cfg.aug_lag_tol_gap and lN_rel < cfg.aug_lag_tol_lN:
                print(f" Multiplier value: min={np.min(self.lN_updated):.4e}, max={np.max(self.lN_updated):.4e}")
                # Accept the classical update and return
                prob.lN_sub.x.array[:] = lN_candidate
                prob.lN_sub.x.scatter_forward()
                # record metrics on solver for external access
                self.last_aug_iters = aug_iter + 1
                if len(newton_its_list) > 0:
                    self.total_newton_its = float(sum(newton_its_list))
                else:
                    self.total_newton_its = 0.0
                return ConvergenceStatus.CONVERGED, aug_iter + 1, None
            
            # Update lN            
            self.lN_updated = lN_candidate
            prob.lN_sub.x.array[:] = self.lN_updated
            prob.lN_sub.x.scatter_forward()


        # Reached maximum augmentation iterations: record metrics
        self.last_aug_iters = cfg.aug_lag_max_iter
        if len(newton_its_list) > 0:
            self.total_newton_its = float(sum(newton_its_list))
        else:
            self.total_newton_its = 0.0

        return ConvergenceStatus.MAX_ITERATIONS, cfg.aug_lag_max_iter, None

    def _solve_augmented_lagrangian(self, d_val, step):
        """Solve contact problem with alternating solution strategy
        and augmented Lagrangian method for contact enforcement.
        """
        cfg, prob = self.cfg, self.problem

        # Alternating parameters
        max_alt_iter = 10 if cfg.use_alternating else 1
        alt_tol = 0.01 if cfg.use_alternating else 10 
        
        print(f"\n{'='*70}")
        print(f"Starting alternating scheme for d={d_val:.4f}")
        print(f"{'='*70}")
        
        # Store previous displacement and xi for convergence check
        u_prev = prob.u.x.array.copy()
        xi_prev = self.xi.x.array.copy() if self.xi is not None else None
        
        total_newton_its = 0

        # Create NonlinearProblem once outside the loop.
        nlp = NonlinearProblem(
            F=prob.residual,
            u=[prob.u, prob.lN],
            J=prob.Jac, 
            bcs=prob.bcs,
            petsc_options=self.petsc_options,
            petsc_options_prefix="contact_sig",
            entity_maps=[prob.map_to_parent]
        )
        # nlp._snes.setMonitor(self.snes_debug_monitor)

        # Quantities on submesh
        self.X_sub = ufl.SpatialCoordinate(prob.contact_mesh)
        self.u_sub = Function(prob.V_sub)
        try:
            self.u_sub.interpolate_nonmatching(prob.u, cells=np.arange(prob.num_sub_cells),
                                            interpolation_data=prob.interp_data)
        except Exception as e:
            print(f"Warning: interpolate_nonmatching u_sub failed: {e}")
            print("Falling back to zero displacement on submesh for postprocessing.")
            pass

        for alt_iter in range(max_alt_iter):
            # ================================================================
            # Step 1: Update ray-tracing mapping (xi solver)
            # ================================================================

            self.xi = ray_tracing_mapping(
                parent_mesh=prob.domain,
                contact_mesh=prob.contact_mesh,
                facet_marker=prob.facet_tags,
                c_tag=prob.ft['contact'],
                metadata=prob.metadata,
                nx=prob.nx,
                x=prob.X + prob.u,
                xi_guess=self.xi if self.xi is not None else None,
                map_to_parent=prob.map_to_parent,
                d_val=d_val,
                plane_loc=cfg.plane_loc,
                R=cfg.R,
                step_idx=step,
                output_dir=os.path.join(cfg.output_dir, "ray_tracing"),
                debug_plots=cfg.debug_plots if alt_iter == 0  else False,  # Only debug first iteration
                debug_monitor=False,
                u_field=self.u_sub,
            )

            
            # Update positions of projected points on reference plane
            y = ufl.as_vector([self.xi, cfg.plane_loc * np.ones_like(self.xi)])

            y_sub = fem.Function(prob.V_sub)
            y_sub.interpolate(Expression(y, prob.V_sub.element.interpolation_points))
            # Update y_func for gap evaluation in contact residual
            prob.y_func.x.array[:] = y_sub.x.array[:]
            prob.y_func.x.scatter_forward()

            # =================================================================================
            # Step 2: Solve contact problem for the displacement field and Lagrange multiplier
            # =================================================================================
            try:
                nlp.solve()
                reason = nlp.solver.getConvergedReason()
                n_its = nlp.solver.getIterationNumber()
                total_newton_its += n_its

                if reason < 0:
                    print(f"  Alt iter {alt_iter+1}: SNES failed with reason {reason}")
                    return ConvergenceStatus.SNES_FAILED, total_newton_its, reason
                
            except Exception as e:
                if self.comm.rank == 0:
                    print(f"  Alt iter {alt_iter+1}: SNES exception: {e}")
                return ConvergenceStatus.EXCEPTION, total_newton_its, None
            
            # ================================================================
            # Step 3: Check convergence
            # ================================================================
            # Convergence requires BOTH displacement AND contact mapping to stabilize
            
            # Displacement-based metrics
            du = prob.u.x.array - u_prev
            du_norm = np.linalg.norm(du)
            u_norm = np.linalg.norm(prob.u.x.array)
            rel_change_u = du_norm / (u_norm + 1e-10)  
            
            # Contact mapping (xi) metrics
            if xi_prev is not None:
                dxi = self.xi.x.array - xi_prev
                dxi_norm = np.linalg.norm(dxi)
                xi_norm = np.linalg.norm(self.xi.x.array)
                rel_change_xi = dxi_norm / (xi_norm + 1e-10)
            else:
                # First iteration: compare with X_dofs
                V_xi = functionspace(prob.contact_mesh, ("CG", 1))
                X_dofs = V_xi.tabulate_dof_coordinates()[:, 0]
                xi_prev = X_dofs
                dxi_norm = np.linalg.norm(self.xi.x.array - xi_prev)
                xi_norm = np.linalg.norm(self.xi.x.array)
                rel_change_xi = dxi_norm / (xi_norm + 1e-10)

            print(f"  Alt iter {alt_iter+1}: Newton its={n_its}, "
                  f"||du||={du_norm:.4e} (rel={rel_change_u:.4e}), "
                  f"||dξ||={dxi_norm:.4e} (rel={rel_change_xi:.4e})")

            # Check convergence: BOTH criteria must be satisfied
            converged_u = rel_change_u < alt_tol
            converged_xi = rel_change_xi < alt_tol and xi_prev is not None # skip first iteration for xi convergence check
            
            if converged_u and converged_xi:
                print(f"  -> Alternating minimization converged in {alt_iter+1} iterations")
                print(f"  -> Total Newton iterations: {total_newton_its}")
        
                return ConvergenceStatus.CONVERGED, total_newton_its, reason

            # Update previous state for next iteration
            u_prev[:] = prob.u.x.array
            xi_prev = self.xi.x.array.copy()

        # Maximum iterations reached
        print(f"  -> Alternating minimization reached max iterations ({max_alt_iter})")
        print(f"  -> Final: rel_u={rel_change_u:.4e}, rel_ξ={rel_change_xi:.4e} (tol={alt_tol:.4e})")
        return ConvergenceStatus.MAX_ITERATIONS, total_newton_its, None

    def _compute_penetration(self, d_val):
        """Compute penetration metrics on the contact surface.

        The gap is evaluated at quadrature points on the parent mesh,
        consistent with the integral enforcement of the impenetrability
        constraint in the variational formulation.

        Returns
        -------
        pen_L2 : float
            Integral norm L²(Γc) of penetration: √(∫ min(g,0)² ds).
        pen_max : float
            Maximum pointwise penetration at quadrature points (mesh-independent).
        """
        prob = self.problem
        cfg = self.cfg

        # ------------------------------------------------------------------
        # 1. Build y_parent on the parent mesh (same pattern as postprocessing)
        # ------------------------------------------------------------------
        # y is already stored on the submesh (prob.y_func); interpolate it
        # back to the parent DG vector space so the gap UFL lives on the
        # parent mesh and can be integrated over ds_top.
        sub_cells = np.arange(prob.num_sub_cells, dtype=np.int32)
        parent_facets = prob.map_to_parent.sub_topology_to_topology(sub_cells, False)
        tdim = prob.domain.topology.dim
        fdim = tdim - 1
        prob.domain.topology.create_connectivity(fdim, tdim)
        facet_to_cell = prob.domain.topology.connectivity(fdim, tdim)
        parent_cells = np.array([facet_to_cell.links(f)[0] for f in parent_facets],
                                dtype=np.int32)

        y_parent = Function(prob.V_parent)
        y_parent.x.array[:] = 0.0
        try:
            interp_sub_to_parent = fem.create_interpolation_data(
                prob.V_parent, prob.V_sub, cells=parent_cells)
            y_parent.interpolate_nonmatching(prob.y_func, cells=parent_cells,
                                             interpolation_data=interp_sub_to_parent)
            y_parent.x.scatter_forward()
        except Exception:
            y_parent.x.array[:] = 0.0
            y_parent.x.scatter_forward()

        # ------------------------------------------------------------------
        # 2. Gap as UFL on parent mesh, integrated over contact boundary
        # ------------------------------------------------------------------
        g_parent = gap_rt(prob.nx, prob.X + prob.u, y_parent)
        ds_contact = prob.ds(prob.ft["contact"])

        # min(g, 0) = penetration (negative part of the gap)
        pen_expr = ufl.min_value(g_parent, 0.0)

        # L2-norm of penetration: sqrt( ∫_Γc min(g,0)² ds )
        pen_L2_sq = self.comm.allreduce(
            fem.assemble_scalar(fem.form(pen_expr**2 * ds_contact)), MPI.SUM)
        pen_L2 = np.sqrt(pen_L2_sq)

        # Max penetration for logging (evaluate at quadrature points)
        pen_max = 0.0
        try:
            contact_facets = prob.facet_tags.find(prob.ft["contact"])
            quad_deg = prob.metadata["quadrature_degree"]
            _, g_quad, _ = expression_at_quadrature(
                prob.domain, g_parent, quad_deg, contact_facets)
            pen_vals = np.minimum(g_quad, 0.0)
            pen_max_local = np.max(np.abs(pen_vals)) if len(pen_vals) > 0 else 0.0
            pen_max = self.comm.allreduce(pen_max_local, MPI.MAX)
        except Exception:
            pass


        if self.comm.rank == 0:
            print(f"  Penetration check: L2-norm={pen_L2:.4e}, max|pen|={pen_max:.4e}")

        return pen_L2, pen_max

    def _solve_snes_vi(self):
        """Solve using SNES VI contact."""
        cfg = self.cfg
        prob = self.problem
        ndim = prob.domain.geometry.dim

        # Setup inequality (contact) constraint
        # The displacement u must be such that the current configuration x+u
        # remains in the box [xmin = -inf,xmax = inf] x [ymin = -g0,ymax = inf]
        # This comes from developing: g_PR = (X + u - Y) · Ny >= 0  =>  u_y >= -g0, with g0=(X - Y)·Ny
        # inf replaced by large number for implementation
        lims = np.zeros(2 * ndim)
        for i in range(ndim):
            lims[2 * i] = -1e7
            lims[2 * i + 1] = 1e7
        lims[-2] = -cfg.plane_loc  

        def _constraint_u(x):
            values = np.zeros((ndim, x.shape[1]))
            for i in range(ndim):
                values[i] = lims[2 * i + 1] - x[i]
            return values

        def _constraint_l(x):
            values = np.zeros((ndim, x.shape[1]))
            for i in range(ndim):
                values[i] = lims[2 * i] - x[i]
            return values

        umax = fem.Function(prob.V_u)
        umax.interpolate(_constraint_u)
        umin = fem.Function(prob.V_u)
        umin.interpolate(_constraint_l)

        nlp = NonlinearProblem(
            F=prob.R_total,
            u=prob.u,
            bcs=prob.bcs,
            petsc_options=self.petsc_options,
            petsc_options_prefix="contact_sig",
            entity_maps=[prob.map_to_parent]
        )
        
        # Set variable bounds 
        nlp.solver.setVariableBounds(umin.x.petsc_vec, umax.x.petsc_vec)

        try:
            u_vec = prob.u.x.petsc_vec.copy()
            nlp.solver.solve(None, u_vec)
            reason = nlp.solver.getConvergedReason()
            n_its = nlp.solver.getIterationNumber()

            prob.u.x.petsc_vec.array[:] = u_vec.array
            prob.u.x.scatter_forward()

            if reason < 0:
                return ConvergenceStatus.SNES_FAILED, n_its, reason

            return ConvergenceStatus.CONVERGED, n_its, reason

        except Exception as e:
            if self.comm.rank == 0:
                print(f"SNES exception: {e}")
            return ConvergenceStatus.EXCEPTION, 0, e

    def snes_debug_monitor(self, snes, its, fgnorm):
        """
        Monitor called at each SNES iteration 
        (if "nlp._snes.setMonitor(self.snes_debug_monitor)" is uncommented).
        --> Compute and print contact-specific metrics for debugging.
        """
        cfg, problem = self.cfg, self.problem
        
        # Compute Jacobian of the deformation 
        adjacent_cells, _, _ = extract_boundary_cells(problem.domain, problem.facet_tags, problem.ft['contact'])
        V_J = functionspace(problem.domain, ("DG", 0))
        if not cfg.linear:
            J_fun = fem.Function(V_J)
            J_fun.interpolate(Expression(problem.J, V_J.element.interpolation_points))
            J_contact = J_fun.x.array[adjacent_cells]
            J_min = J_contact.min()
            J_max = J_contact.max()
            print(f"  Jacobian range: [{J_min:.4f}, {J_max:.4f}]")
            if J_min < 0.1:
                print(f"  WARNING: Severe mesh distortion detected!")
        
# =============================================================================
# RESULTS
# =============================================================================

@dataclass
class Results:
    """Container for simulation results."""
    step: List[int] = field(default_factory=list)
    d_val: List[float] = field(default_factory=list)
    Fn: List[float] = field(default_factory=list)
    x_quad: List[np.ndarray] = field(default_factory=list)
    p_quad: List[np.ndarray] = field(default_factory=list)
    x_quad_g: List[np.ndarray] = field(default_factory=list)
    g_quad: List[np.ndarray] = field(default_factory=list)
    newton_its: List[int] = field(default_factory=list)
    aug_iters: List[int] = field(default_factory=list)
    avg_newton_its: List[float] = field(default_factory=list)
    pc_error_L2: List[float] = field(default_factory=list)
    x_nodes: List[np.ndarray] = field(default_factory=list)
    
    def save(self, filename: str):
        np.savez(filename, **{k: np.array(v) for k, v in self.__dict__.items()})

# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation(cfg: SimulationConfig):
    """Run the Signorini contact simulation."""
    
    comm = MPI.COMM_WORLD
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    if comm.rank == 0:
        print("=" * 60)
        print("Signorini Contact: Compressible Disk on Rigid Plane")
        print("RAY-TRACING mapping for gap function")
        print("=" * 60)
        print(f"Material: {cfg.hyperelastic_model if not cfg.linear else 'Linear Elastic'}, Y={cfg.Y:.2e}, nu={cfg.nu}")
        print(f"Contact method: {cfg.contact_method}")
        if cfg.contact_method == "nitsche":
            print(f"  Nitsche parameters: θ={cfg.theta}, γ₀={cfg.gamma0}")
        elif cfg.contact_method == "uzawa":
            print(f"  Uzawa penalty={cfg.uzawa_penalty:.2e}")
        elif cfg.contact_method == "augmented_lagrangian":
            print(f"  Aug.Lag. penalty={cfg.aug_lag_penalty:.2e}")
        print("="*60)
    

    # Save simulation parameters for postprocessing
    if comm.rank == 0:
        sim_params_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        np.savez(os.path.join(cfg.output_dir, f"simulation_params_P{cfg.u_degree}.npz"), **sim_params_dict)

    # Initialize problem and solver
    problem = ContactProblem(cfg)
    solver = ContactSolver(problem, cfg)
    results = Results()
    
    # Output
    vtx_u = VTXWriter(comm, os.path.join(cfg.output_dir, f"solution_u.bp"), [problem.u], engine="BP4")
    vtx_sigma = VTXWriter(comm, os.path.join(cfg.output_dir, f"solution_sigma.bp"), [problem.sigma_DG], engine="BP4")

    # Create load steps (skip d=0)
    d_values = np.linspace(0, cfg.d_final, cfg.n_steps + 1)[1:]
   
    if comm.rank == 0:
        print(f"\nStarting simulation: {cfg.n_steps} steps, d_final={cfg.d_final}")
        print("-" * 60)

    for step, d_val in enumerate(d_values):
        if comm.rank == 0:
            print(f"\nStep {step + 1}/{cfg.n_steps}: d = {d_val:.4f}")
        
        status, n_its, reason = solver.solve_step(d_val, step + 1)
        
        # Ensure convergence        
        if status != ConvergenceStatus.CONVERGED:
            print(f"  SNES failed with reason: {reason}, its={n_its}")
            break
        
        # Ensure solver exposes augmentation metrics for non-augmented runs
        if cfg.contact_method != "uzawa":
            solver.last_aug_iters = 0
            # average Newton its is simply the Newton its for single solve
            try:
                solver.total_newton_its = float(n_its)
            except Exception:
                solver.total_newton_its = 0.0
    
        # =============================================================================
        # POSTPROCESSING
        # =============================================================================

        # Compute normal force (vertical component of traction vector integrated over contact boundary)
        ds_contact = problem.ds(problem.ft["contact"])
        quad_deg = problem.metadata["quadrature_degree"]
        contact_facets = problem.facet_tags.find(problem.ft['contact'])
        sigma_nn = problem.P_n # using first Piola-Kirchhoff stress
        x_quad, sigma_nn_quad, _ = expression_at_quadrature(
                problem.domain, sigma_nn, quad_deg, contact_facets) 

        if cfg.contact_method == "nitsche" or cfg.contact_method == "snes_vi":
            Fn_form = fem.form(- ufl.dot(problem.P_N, - problem.Ny) * ds_contact) 
            Fn = comm.allreduce(fem.assemble_scalar(Fn_form), op=MPI.SUM)
            p_quad = -sigma_nn_quad  # Contact pressure at quadrature points on contact boundary
        
        elif cfg.contact_method == "augmented_lagrangian":
            Fn_form = fem.form(-problem.lN * ds_contact, entity_maps=[problem.map_to_parent])
            Fn = comm.allreduce(fem.assemble_scalar(Fn_form), op=MPI.SUM)  
            # LM at quadrature points on reference configuration
            lN_quad = Function(problem.V_q, name="lN_quad")
            lN_quad.interpolate(Expression(problem.lN, problem.V_q.element.interpolation_points))
            lN_quad.x.scatter_forward()
            p_quad = - lN_quad.x.array  
        
        elif cfg.contact_method == "uzawa":
            Fn_form = fem.form(-problem.lN_sub * ds_contact, entity_maps=[problem.map_to_parent])
            Fn = comm.allreduce(fem.assemble_scalar(Fn_form), op=MPI.SUM)   
            # LM at quadrature points on reference configuration
            lN_quad = Function(problem.V_q, name="lN_quad")
            lN_quad.interpolate(Expression(problem.lN_sub, problem.V_q.element.interpolation_points))
            lN_quad.x.scatter_forward()
            p_quad = - lN_quad.x.array 
       
        # Project Cauchy stress onto DG space for output 
        try:
            problem.sigma_DG.interpolate(Expression(problem.sigma, problem.V_stress.element.interpolation_points))
            problem.sigma_DG.x.scatter_forward()
        except Exception as e:
            print(f"Warning: interpolate Expression for sigma_DG failed: {e}; zeroing out")
            problem.sigma_DG.x.array[:] = 0.0
            problem.sigma_DG.x.scatter_forward()

        if cfg.contact_method == "snes_vi":
            # In this case, we have to use the g^PR that measures the vertical gap
            g0 = ufl.dot(problem.Ny, problem.X - problem.y)
            gap_expr = g0 + ufl.dot(problem.Ny, problem.u)    
            x_quad_g, g_quad, _ = expression_at_quadrature(
                problem.domain, gap_expr, quad_deg, contact_facets)

        else:        
            # Compute gap by first interpolating `y_func` (on submesh) to the
            # parent mesh, so the gap expression lives on a single domain
            # (the parent mesh). 
            try:
                # Map submesh cell indices -> parent facet indices
                sub_cells = np.arange(problem.num_sub_cells, dtype=np.int32)
                parent_facets = problem.map_to_parent.sub_topology_to_topology(sub_cells, False)
                # Build facet->cell connectivity to get parent cells adjacent to facets
                tdim = problem.domain.topology.dim
                fdim = tdim - 1
                problem.domain.topology.create_connectivity(fdim, tdim)
                facet_to_cell = problem.domain.topology.connectivity(fdim, tdim)
                parent_cells = np.array([facet_to_cell.links(f)[0] for f in parent_facets], dtype=np.int32)

                # Interpolate y_func (submesh) to parent vector space
                y_parent = Function(problem.V_parent)
                y_parent.x.array[:] = 0.0   
                try:
                    # Create interpolation data for sub->parent mapping for these cells
                    interp_sub_to_parent = fem.create_interpolation_data(
                        problem.V_parent, problem.V_sub, cells=parent_cells)
                    y_parent.interpolate_nonmatching(problem.y_func, cells=parent_cells,
                                                        interpolation_data=interp_sub_to_parent)
                    y_parent.x.scatter_forward()
                except Exception:
                    # Fallback: zero out y_parent   
                        y_parent.x.array[:] = 0.0
                        y_parent.x.scatter_forward()

                # Now build gap on parent mesh and evaluate at quadrature points on contact boundary
                g_parent = gap_rt(problem.nx, problem.X + problem.u, y_parent)
                x_quad_g, g_quad, _ = expression_at_quadrature(
                    problem.domain, g_parent, quad_deg, contact_facets)

            except Exception as e:
                print(f"Warning: gap evaluation at quadrature points failed: {e}; setting gap to zero")
                # If there is an exception, produce a zero gap array
                g_quad = np.zeros_like(p_quad)

        # Nodal positions on contact submesh in current configuration
        X_sub = ufl.SpatialCoordinate(problem.contact_mesh)
        u_sub = Function(problem.V_sub)
        try:
            u_sub.interpolate_nonmatching(problem.u, cells=np.arange(problem.num_sub_cells),
                                            interpolation_data=problem.interp_data)
        except Exception as e:
            print(f"Warning: interpolate_nonmatching for u_sub failed: {e}; using fallback zeros")
            u_sub.x.array[:] = 0.0
            u_sub.x.scatter_forward()
        
        x_current = fem.Function(problem.V_sub)
        x_current.interpolate(fem.Expression(X_sub + u_sub, problem.V_sub.element.interpolation_points))
        x_current.x.scatter_forward()

        # L2-error of contact pressure against Hertz solution 
        if cfg.contact_method in ["nitsche", "snes_vi"]:
            x_H = problem.X            
            pc_fem = - sigma_nn                         
        else:
            x_H = solver.X_sub
            pc_fem = - problem.lN if cfg.contact_method == "augmented_lagrangian" else - problem.lN_sub 

        E_star = cfg.Y / (1 - cfg.nu ** 2)          # Effective Young's modulus
        a = np.sqrt(4 * Fn * cfg.R / (np.pi * E_star)) # Contact half-width from Hertz theory
        pc_H = 2 * Fn / (ufl.pi * a) * ufl.sqrt(       # Hertz pressure distribution, zero outside contact area
            ufl.max_value(0.0, 1 - (x_H[0] / a)**2)
        )
        
        pc_error_L2 = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((pc_fem - pc_H) ** 2 * ds_contact, entity_maps=[problem.map_to_parent])), op=MPI.SUM
        )
    )
        pc_H_norm = np.sqrt(
            comm.allreduce(
                fem.assemble_scalar(fem.form(pc_H ** 2 * ds_contact, entity_maps=[problem.map_to_parent])), op=MPI.SUM
            )
        )
        eps = 1e-12
        pc_rel_error_L2 = pc_error_L2 / (pc_H_norm + eps)
        if comm.rank == 0:
            print(f"Relative L2-error: {pc_rel_error_L2 * 100:.2f}%")  
       
        # Monitoring
        if comm.rank == 0:
            print("--- Monitoring ---")
            print(f"Elastic energy: {fem.assemble_scalar(fem.form(problem.psi * problem.dx)):.4e}")
            print(f"Normal contact force Fn: {Fn:.4e} N")
            print(f"Contact pressure (quad points): min: {p_quad.min()}, max: {p_quad.max()}")
            print(f"Gap function (quad points): min: {g_quad.min()}, max: {g_quad.max()}")

        # Store results
        results.step.append(step + 1)
        results.d_val.append(d_val)
        results.Fn.append(Fn)
        results.x_quad.append(x_quad)
        results.p_quad.append(p_quad)
        results.x_quad_g.append(x_quad_g)
        results.g_quad.append(g_quad)
        results.newton_its.append(n_its)
        results.aug_iters.append(solver.last_aug_iters)
        results.avg_newton_its.append(solver.total_newton_its)
        # results.pc_error_L2.append(pc_rel_error_L2)
        results.x_nodes.append(x_current.x.array.copy())

        # Save results at each step
        results.save(os.path.join(cfg.output_dir, f"results_P{cfg.u_degree}.npz"))
        
        # Write output
        vtx_u.write(float(step + 1))
        vtx_sigma.write(float(step + 1))
        
        # Store state
        problem.store_state()
    
    # Cleanup
    vtx_u.close()
    vtx_sigma.close()
    
    if status == ConvergenceStatus.CONVERGED and comm.rank == 0:
        print("\n" + "=" * 60)
        print("Simulation complete!")
        print(f"Results saved to {cfg.output_dir}")
        print("=" * 60)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signorini contact problem: compressible hyperelastic half-disk on rigid plane"
    )
    parser.add_argument(
        "--contact-method",
        type=str,
        choices=["nitsche", "augmented_lagrangian", "uzawa", "snes_vi"],
        default="nitsche",
        help="Contact enforcement method (default: nitsche)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: output_signorini_nitsche)"
    )
    parser.add_argument(
        "--gamma0", type=float, default=None,
        help="Nitsche stabilization parameter gamma=gamma0*Y/h (default: from config)"
    )
    parser.add_argument(
        "--theta", type=int, default=None, choices=[-1, 0, 1],
        help="Nitsche parameter for variant selection: 1=symmetric, -1=skew, 0=unsymmetric"
    )
    parser.add_argument(
        "--lcmin", type=float, default=None,
        help="Minimum mesh size in contact region (default: 0.01)"
    )
    parser.add_argument(
        "--u-degree", type=int, default=None,
        help="Displacement polynomial degree (default: from config)"
    )
    
    args = parser.parse_args()
    
    cfg = SimulationConfig()
    cfg.contact_method = args.contact_method
    
    if args.gamma0 is not None:
        cfg.gamma0 = args.gamma0
    if args.theta is not None:
        cfg.theta = args.theta
    if args.lcmin is not None:
        cfg.LcMin = args.lcmin
    if args.u_degree is not None:
        cfg.u_degree = args.u_degree
    if args.output is not None:
        cfg.output_dir = args.output
    
    # Recompute derived parameters (penalty, tolerances, etc.) after CLI overrides
    cfg.recompute_derived()
        
    run_simulation(cfg)
