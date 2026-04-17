#!/usr/bin/env python3
"""
Hyperelastic Contact: Rigid Indenter on Symmetric Half-Space
============================================================

Features:
- Displacement-only formulation (compressible)
- Symmetric mesh (elastic_block_sym_DI)
- Alternating minimization for ray-tracing + displacement
- Multiple contact methods (Nitsche, Augmented Lagrangian)
- Nonlinearity metrics (Green-Lagrange E vs small strain eps)
- VTX output
"""

import os, signal, pickle, shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import argparse

import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import fem, mesh, default_scalar_type
import dolfinx
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.fem import Constant, Function, Expression, functionspace
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix
import basix
from basix import CellType

from utils import (
    elastic_block_sym_DI,
    elastic_block_DI,
    compute_kinematics,
    get_constitutive_model,
    compute_stress,
    compute_stress_linear,
    pneg,
    create_contact_submesh,
    gap_rt,
    extract_boundary_cells,
    expression_at_quadrature,
    compute_normal_force,
    project_vector_on_boundary,
    ray_tracing_sliding,
    CheckpointManager,
)

# =============================================================================
# CONVERGENCE STATUS ENUM
# =============================================================================

class ConvergenceStatus(Enum):
    CONVERGED = 0
    MAX_ITERATIONS = 1
    SNES_FAILED = 2
    EXCEPTION = 4

# ── SNES convergence-reason look-up (PETSc codes) ──────────────────────
SNES_REASON_NAMES = {
    # /* converged */
    2: "CONVERGED_FNORM_ABS",         # ||F|| < atol
    3: "CONVERGED_FNORM_RELATIVE",    # ||F|| < rtol*||F_initial||  
    4: "CONVERGED_SNORM_RELATIVE",    # Newton computed step size small; || delta x || < stol || x ||
    5: "CONVERGED_ITS",               # maximum iterations reached 
    7: "CONVERGED_USER",              # The user has indicated convergence for an arbitrary reason
    # /* diverged */
   -1: "DIVERGED_FUNCTION_DOMAIN",    # The new x location passed the function is not in the domain of F
   -2: "DIVERGED_FUNCTION_COUNT",
   -3: "DIVERGED_LINEAR_SOLVE",       # Linear solver failed to converge
   -4: "DIVERGED_FNORM_NAN",
   -5: "DIVERGED_MAX_IT",
   -6: "DIVERGED_LINE_SEARCH",        # Line search failed to converge
   -7: "DIVERGED_INNER",              # Inner solve failed 
   -8: "DIVERGED_LOCAL_MIN",          # || J^T b || is small, implies convergence to a local minimum of F()
   -9: "DIVERGED_DTOL",               # || F || > divtol * || F_initial ||
    0: "CONVERGED_ITERATING",
}

@dataclass
class SimulationConfig:
    # Indenter Geometry
    R: float = 1.0
    h0: float = 0.0
    ind_center: Tuple[float, float] = (0.0, 0.0)
    y_cap: float = 4.0
    
    # Half-space Geometry
    Lx: float = 10.0
    Ly: float = 10.0
    
    # Loading
    d_final: float = 1.0
    n_steps_indent: int = 10

    # Mesh parameters
    symmetric: bool = True
    a: float = np.sqrt(2 * R * d_final)  # approximate contact radius
    LcMin: float = 0.1 
    refinement_ratio: int = 80 
    Dmin: float = 1.1 * a
    Dmax: float = (1 + 0.1) * Dmin
    
    # Material
    linear: bool = False             # If True, use linear elasticity instead of hyperelasticity
    hyperelastic_model: str = "NH"
    Y: float = 1.0  # MPa
    nu: float = 0.3
    
    # FE degree
    u_degree: int = 1
    lN_degree: int = 1

    # Quadrature degree
    quad_degree_c: int = 4 # 2, 4, 6, 8
    quad_degree_v: int = 4
    
    # Contact method
    contact_method: str = "nitsche"
    
    # Nitsche parameters
    nitsche_theta: int = 0
    nitsche_gamma0: float = 1.0
    gamma_adaptive: bool = False          # Nitsche: adaptive γ₀ strategy
    gamma0_estimate_current: bool = True  # If False, estimate γ₀ using the linearised elasticity eigenvalue problem 

    # Augmented Lagrangian (with and without Uzawa) contact parameters
    al_coeff: float = 1.0    # Adimensional coefficient for scaling
    uzawa_penalty: float = al_coeff * Y / LcMin
    aug_lag_penalty: float = al_coeff * Y / 1.0 # for dimensional consistency, use LcMin=1.0 here
    aug_lag_max_iter: int = 50
    aug_lag_tol_gap: float = 1e-2 * LcMin # % of the minimum mesh size
    aug_lag_tol_lN: float = 1e-2          # Tolerance for Lagrange multiplier convergence (relative change)
    
    # Solver
    max_iter: int = 50
    snes_rtol: float = 1e-9
    snes_atol: float = 1e-9
    snes_linesearch: str = "bt"       # SNES line search type
    max_step_cuts: int = 4
    use_alternating: bool = False      # Whether to use alternating scheme for ray-tracing + displacement
    max_alt_iter: int = 10
    
    # Alternating scheme: ξ relaxation
    xi_relax: float = 1.0             # ξ under-relaxation factor ∈ (0,1]; 1.0 = no damping
    
    # Diagnostics
    diagnose: bool = False            # Print detailed per-step diagnostics
    
    # Output
    output_dir: str = "output_indentation_" + contact_method
    checkpoint_interval: int = 5
    debug_plots: bool = True
    
    def recompute_derived(self):
        """Recompute parameters derived from base settings (call after CLI overrides)."""
        self.a = np.sqrt(2 * self.R * self.d_final)
        self.Dmin = 1.1 * self.a
        self.Dmax = (1 + 0.1) * self.Dmin
        self.uzawa_penalty = self.al_coeff * self.Y / self.LcMin
        self.aug_lag_tol_gap = 1e-2 * self.LcMin

class ContactProblem:
    """
    Indentation of rigid parabolic indenter on hyperelastic plane.

    Compressible formulation (displacement only, no pressure multiplier).
    """
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.comm = MPI.COMM_WORLD
        self._setup_mesh()
        self._setup_spaces()
        self._setup_material()
        self._setup_contact()
    
    def _setup_mesh(self):
        cfg = self.cfg
        self.h_min = cfg.LcMin
        self.ft = {"left": 1, "right": 2, "top": 3, "bottom": 4}
        
        if cfg.symmetric:
            self.domain, _, self.facet_tags = elastic_block_sym_DI(
                cfg.ind_center, cfg.Dmin, cfg.Dmax, self.ft, {"all": 100},
                self.h_min, cfg.Lx, cfg.Ly, cfg.refinement_ratio)
        
        else: 
            # Full domain (no symmetry)
            self.domain, _, self.facet_tags = elastic_block_DI(
                cfg.ind_center, cfg.Dmin, cfg.Dmax, self.ft, {"all": 100},
                self.h_min, cfg.Lx, cfg.Ly, cfg.refinement_ratio)
        
        ndim = self.domain.topology.dim
        # Create contact submesh
        self.contact_mesh, self.map_to_parent, self.V_sub_, self.V_parent_, self.interp_data_vec, self.num_sub_cells = \
            create_contact_submesh(self.domain, self.facet_tags, self.ft["top"], 1, ndim)
        
        if self.comm.rank == 0:
            n_cells = self.domain.topology.index_map(self.domain.topology.dim).size_global
            print(f"Mesh: {n_cells} cells, h_min = {self.h_min:.3e}")
            print(f"Mesh type: {'SYMMETRIC' if cfg.symmetric else 'FULL'} indentation mesh")
            # Save mesh for inspection
            os.makedirs(cfg.output_dir, exist_ok=True)
            mesh_path = os.path.join(cfg.output_dir, "mesh.xdmf")
            with XDMFFile(self.comm, mesh_path, "w") as xdmf:
                xdmf.write_mesh(self.domain)
                xdmf.write_meshtags(self.facet_tags, self.domain.geometry)
            print(f"Mesh saved to {mesh_path}")

            print(f"Displacement formulation: P{cfg.u_degree} (compressible)")
            print(f"Material: {cfg.hyperelastic_model}")
            print(f"Young's modulus Y: {cfg.Y:.2e}, Poisson's ratio ν: {cfg.nu:.2f}")
            print(f"Contact method: {cfg.contact_method}")
            if cfg.contact_method == "nitsche":
                print(f"  Nitsche parameters: θ={cfg.nitsche_theta}, γ₀={cfg.nitsche_gamma0}")
            elif cfg.contact_method == "uzawa":
                print(f"  Uzawa penalty={cfg.uzawa_penalty:.2e}")
            elif cfg.contact_method == "augmented_lagrangian":
                print(f"  Aug.Lag. penalty={cfg.aug_lag_penalty:.2e}")

    def _setup_spaces(self):
        """Create function spaces and measures."""
        cfg = self.cfg
        ndim = self.domain.topology.dim

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
        self.dx = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": cfg.quad_degree_v})
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags,
                             metadata=self.metadata)
        
        # Other Contact spaces for post-processing and interpolation
        # DG0 for normals
        self.V_n = functionspace(self.domain, ("DG", 0, (2,)))
        self.V_n_sub = functionspace(self.contact_mesh, ("DG", 0, (2,)))
        self.interpolation_data_n = fem.create_interpolation_data(
            self.V_n_sub, self.V_n, cells=np.arange(self.num_sub_cells)
        )

        # Scalar spaces for Lagrange multipliers (augmented Lagrangian) or gap function evaluation
        self.V_scalar = functionspace(self.domain, ("CG", 1))
        self.V_sub_scalar = functionspace(self.contact_mesh, ("CG", 1))
        self.interp_data_scalar = fem.create_interpolation_data(
            self.V_sub_scalar, self.V_scalar, cells=np.arange(self.num_sub_cells))
        
        # Quadrature space on contact boundary for LM post-processing
        q_element = basix.ufl.quadrature_element(
            self.contact_mesh.basix_cell(), value_shape=(), scheme="default", degree=cfg.quad_degree_c
        )
        self.V_q = functionspace(self.contact_mesh, q_element)

        # Dirichlet BCs
        # Fix bottom edge
        fdim = self.domain.geometry.dim - 1
        bottom_dofs = fem.locate_dofs_topological(
            self.V_u, fdim, self.facet_tags.find(self.ft["bottom"]))
        zero_u = Function(self.V_u)
        zero_u.x.array[:] = 0.0
        self.bcs = [fem.dirichletbc(zero_u, bottom_dofs)]
        # Left symmetry --- ux = 0
        if cfg.symmetric:
            left_dofs = fem.locate_dofs_topological(
                self.V_u.sub(0), fdim, self.facet_tags.find(self.ft["left"]))
            self.bcs.append(fem.dirichletbc(default_scalar_type(0.0), left_dofs, self.V_u.sub(0)))

    def _setup_material(self):
        """Setup large deformation kinematics."""
        cfg = self.cfg
        
        # Get kinematics from constitutive.py
        self.kin = compute_kinematics(self.u, self.domain.geometry.dim, cfg.linear)
        
        self.F, self.J, self.C = self.kin['F'], self.kin['J'], self.kin['C']
        self.E = self.kin['E']      # Green-Lagrange strain

        # Get strain energy from constitutive.py (compressible)
        if cfg.linear:
            cfg.hyperelastic_model = "LINEAR"  # Override to use linear elastic model in constitutive.py
        
        self.psi, self.psi_iso, self.psi_vol, self.mat_params = get_constitutive_model(
            self.cfg.hyperelastic_model, self.domain, self.kin, self.cfg.Y, self.cfg.nu)
        
        # Stress measures
        if cfg.linear:
            self.P, self.S, self.sigma = compute_stress_linear(self.psi, self.mat_params['eps']) # P == S == sigma for linear elasticity
                
        else:
            self.P, self.S, self.sigma = compute_stress(self.psi, self.F, self.J)

        # Residual: momentum balance only 
        self.R_elastic = ufl.inner(self.P, ufl.grad(self.du)) * self.dx
    
    def _setup_contact(self):
        cfg = self.cfg

        # Elastic (deformable) half-space --> Slave body
        self.X = ufl.SpatialCoordinate(self.domain)
        self.Nx = ufl.as_vector([0, 1])
        
        self.cofactor = self.J * ufl.inv(self.F).T
        nx_ = ufl.dot(self.cofactor, self.Nx)
        self.nx = nx_ / ufl.sqrt(ufl.dot(nx_, nx_))
        
        if cfg.linear:
            # For linear elasticity, current normal == reference normal
            self.nx = self.Nx    
        
        # Normal component of traction (contact pressure) from 1st Piola–Kirchhoff stress
        self.P_n = ufl.dot(ufl.dot(self.P, self.Nx), self.nx)
        
        # Characteristic mesh size
        self.h = ufl.CellDiameter(self.domain)
        
        # Measure on (top) contact boundary 
        ds_top = self.ds(self.ft["top"])
        
        # Function to hold current y position on contact boundary (for gap evaluation) 
        self.y_func = Function(self.V_sub_, name="y_indenter")
        # Gap function for ray-tracing mapping
        g = gap_rt(self.nx, self.X + self.u, self.y_func)

        # Ray-tracing projection xi placeholders (will be filled by solver)
        try:
            self.V_xi = functionspace(self.contact_mesh, ("CG", cfg.u_degree))
            self.xi = Function(self.V_xi, name="xi")
            self.xi_old = Function(self.V_xi, name="xi_old")
            try:
                self.xi.interpolate(lambda coords: coords[0])
                self.xi_old.interpolate(lambda coords: coords[0])
            except Exception:
                self.xi.x.array[:] = 0.0
                self.xi_old.x.array[:] = 0.0
            self.xi.x.scatter_forward()
            self.xi_old.x.scatter_forward()
        except Exception:
            pass
        
        if cfg.contact_method == "nitsche":
            """ Nitsche's method for contact enforcement """
            # γ₀ as a fem.Constant so it can be updated at runtime
            # (needed for γ₀-continuation without form recompilation)
            self.gamma0_const = Constant(self.domain, default_scalar_type(cfg.nitsche_gamma0))
            
            if cfg.gamma_adaptive:
                # Use max eigenvalue estimate for γ = γ₀ · λ_max
                # lam_max_init = self.estimate_lmbda_max_contact()
                lam_max_init = self.estimate_lmbda_max_contact_local()
                self.lam_max = fem.Constant(self.domain, default_scalar_type(lam_max_init))
                self.gamma = self.gamma0_const * self.lam_max
            else:
                # Use fixed γ = γ₀ * Y / h
                self.gamma = self.gamma0_const * cfg.Y / self.h
            
            P_n_variation = ufl.derivative(self.P_n, self.u, self.du)
            term1 = -(cfg.nitsche_theta * self.P_n * P_n_variation / self.gamma) * ds_top

            c_arg = self.P_n + self.gamma * g
            c_arg_var = ufl.derivative(cfg.nitsche_theta * self.P_n + self.gamma * g, self.u, self.du)
            term2 = (pneg(c_arg) * c_arg_var / self.gamma) * ds_top
            
            self.R_contact = term1 + term2
            self.R_total = self.R_elastic + self.R_contact
        
        elif cfg.contact_method == "uzawa":
            """ Augmented Lagrangian with Uzawa's augmentation loop contact formulation"""
            
            self.eps_contact = cfg.uzawa_penalty # Augmentation parameter 

            self.lN_sub = Function(self.V_sub_scalar, name="lN_sub") # Lagrange multiplier for contact pressure (defined on submesh)
                   
            self.contact_arg = self.lN_sub + self.eps_contact * g
            g_variation = ufl.derivative(g, self.u, self.du)
            
            self.R_contact = pneg(self.contact_arg) * g_variation * ds_top
            self.R_total = self.R_elastic + self.R_contact
        
        elif cfg.contact_method == "augmented_lagrangian":

            self.eps_contact = cfg.aug_lag_penalty # Augmentation parameter 
            
            self.contact_arg = self.lN + self.eps_contact * g
            # g_variation = ufl.dot(self.nx, self.du)  
            g_variation = ufl.derivative(g, self.u, self.du)  # More general variation of the gap function

            self.R_elastic += pneg(self.contact_arg) * g_variation * ds_top
            self.R_elastic += (pneg(self.contact_arg) - self.lN) * self.dlN / self.eps_contact * ds_top 
            self.residual = ufl.extract_blocks(self.R_elastic)
            # define the jacobian
            jac = ufl.derivative(self.R_elastic, self.u, self.delta_u) + ufl.derivative(self.R_elastic, self.lN, self.delta_lN)
            self.Jac = ufl.extract_blocks(jac)  
        else:
            raise ValueError(f"Unknown contact method: {cfg.contact_method}")

    def estimate_lmbda_max_contact_local(self):
        """Estimate the largest Nitsche inverse-inequality eigenvalue via
        element-local generalized eigenvalue problems.

        For every cell K adjacent to the contact boundary, the method
        assembles the (element-level) generalized eigenvalue problem:

            A_K  x = λ_K  B_K  x

        and returns

            λ_max = max_K  λ_K^{max}

        Because each element's contribution must be isolated, the forms are
        assembled in a DG (Discontinuous Galerkin) function space of the
        same polynomial degree.  This makes the global matrices block-diagonal
        so that extracting the per-element blocks is trivial.

        Bilinear forms
        ~~~~~~~~~~~~~~
        *linearised elasticity* (``current_config = False``):
            A_K = ∫_{∂K ∩ Γc} (σ(u)·N)·(σ(v)·N) ds
            B_K = ∫_K          σ(u):ε(v)           dx

        *current-config tangent* (``current_config = True``):
            A_K = ∫_{∂K ∩ Γc} (dP·N)·(dP·N) ds
            B_K = ∫_K          dP : ∇v        dx

        where dP = ∂P/∂u is the tangent of the first Piola–Kirchhoff stress
        evaluated at the current displacement.

        The element stiffness B_K may be singular (rigid-body modes).  The
        solver projects out the null space of B_K before solving the reduced
        eigenproblem.

        Returns
        -------
        lam_max : float
            Largest eigenvalue among all contact-adjacent elements eigenvalues.
        """
        from scipy.linalg import eigh as sp_eigh

        cfg = self.cfg
        tdim = self.domain.topology.dim
        Y, nu = cfg.Y, cfg.nu
        ft = self.ft
        ds_c = self.ds(ft['top'])
        dx = self.dx

        # ── DG space: element DOFs are fully decoupled ────────────────
        V_dg = functionspace(self.domain, ("DG", cfg.u_degree, (2,)))
        u_trial = ufl.TrialFunction(V_dg)
        v_test = ufl.TestFunction(V_dg)

        current_config = cfg.gamma0_estimate_current
        if current_config:
            # --- tangent forms at the current (nonlinear) configuration ---
            dP_u = ufl.derivative(self.P, self.u, u_trial)
            dP_v = ufl.derivative(self.P, self.u, v_test)

            dt_u = ufl.dot(dP_u, self.Nx)
            dt_v = ufl.dot(dP_v, self.Nx)

            A_form = fem.form(ufl.dot(dt_u, dt_v) * ds_c)
            B_form = fem.form(ufl.inner(dP_u, ufl.grad(v_test)) * dx)
        else:
            # --- linearised (small-strain) elasticity ---
            mu = Y / (2 * (1 + nu))
            lmbda = Y * nu / ((1 + nu) * (1 - 2 * nu))
            I = ufl.Identity(tdim)

            def eps(w):
                return ufl.sym(ufl.grad(w))

            def sigma(w):
                return lmbda * ufl.tr(eps(w)) * I + 2 * mu * eps(w)

            n = self.Nx
            t_u = ufl.dot(sigma(u_trial), n)
            t_v = ufl.dot(sigma(v_test), n)
            A_form = fem.form(ufl.inner(t_u, t_v) * ds_c)
            B_form = fem.form(ufl.inner(sigma(u_trial), eps(v_test)) * dx)

        # ── assemble block-diagonal global matrices ───────────────────
        A = fem.petsc.assemble_matrix(A_form)
        A.assemble()
        B = fem.petsc.assemble_matrix(B_form)
        B.assemble()

        # ── identify cells touching the contact boundary ──────────────
        self.domain.topology.create_connectivity(tdim - 1, tdim)
        facet2cell = self.domain.topology.connectivity(tdim - 1, tdim)

        contact_facets = self.facet_tags.find(ft["top"])
        contact_cells = np.unique(
            np.hstack([facet2cell.links(f) for f in contact_facets])
        )

        if contact_cells.size == 0:
            raise RuntimeError("No contact facets/cells found for the provided tag.")

        bs = V_dg.dofmap.bs

        # ── element-local eigenvalue solves ───────────────────────────
        lam_max_local = -np.inf
        n_cells_ok = 0
        null_tol = 1e-10  # relative tolerance for B null-space detection

        for cell in contact_cells:
            # DG DOFs for this cell (block → expanded)
            block_dofs = V_dg.dofmap.cell_dofs(int(cell))
            cell_dofs = np.concatenate(
                [block_dofs * bs + c for c in range(bs)]
            ).astype(np.int32)
            # print(f"Cell {cell}: block DOFs {block_dofs}, expanded DOFs {cell_dofs}")
            nd = len(cell_dofs)

            # Extract local dense matrices from PETSc rows
            A_loc = np.zeros((nd, nd))
            B_loc = np.zeros((nd, nd))
            for i_loc, i_g in enumerate(cell_dofs):
                a_cols, a_vals = A.getRow(int(i_g))
                b_cols, b_vals = B.getRow(int(i_g))
                for j_loc, j_g in enumerate(cell_dofs):
                    # PETSc returns sorted columns in CSR order
                    ia = np.searchsorted(a_cols, j_g)
                    if ia < len(a_cols) and a_cols[ia] == j_g:
                        A_loc[i_loc, j_loc] = a_vals[ia]
                    ib = np.searchsorted(b_cols, j_g)
                    if ib < len(b_cols) and b_cols[ib] == j_g:
                        B_loc[i_loc, j_loc] = b_vals[ib]

            # Skip cells whose contact-facet contribution is zero --- maybe a cell is included 
            # by connectivity but has zero intersection with the contact boundary/no contribution to A_loc.
            if np.allclose(A_loc, 0.0):
                continue

            # --- project out null space of B_loc ---> remove rigid-body modes from each cell 
            eigvals_B, eigvecs_B = sp_eigh(B_loc)  # returns eigenvalues in ascending order
            B_max = max(abs(eigvals_B[-1]), 1e-30)
            keep = eigvals_B > null_tol * B_max  # keep only non-zero eigenvalues
            if not keep.any():
                continue  # B entirely zero on this cell

            Q = eigvecs_B[:, keep]
            # Reduced problem:  Q^T A Q  x = λ  Q^T B Q  x --- project onto the positive-definite subspace of B_loc
            A_red = Q.T @ A_loc @ Q
            B_red = Q.T @ B_loc @ Q  # diagonal-dominant, positive definite

            try:
                eigvals_red = sp_eigh(A_red, B_red, eigvals_only=True)
                lam_cell = float(eigvals_red[-1])
                if lam_cell > lam_max_local:
                    lam_max_local = lam_cell
                n_cells_ok += 1
            except np.linalg.LinAlgError:
                # Fallback: skip cell if factorisation fails
                continue

        # ── MPI reduction ─────────────────────────────────────────────
        lam_max_global = self.comm.allreduce(lam_max_local, MPI.MAX)
        n_cells_total = self.comm.allreduce(n_cells_ok, MPI.SUM)

        if self.comm.rank == 0:
            mode_str = ("current-config tangent" if current_config
                        else "linearised elasticity")
            print(f"  [eig-local] mode: {mode_str}")
            print(f"  [eig-local] {len(contact_cells)} contact cells, "
                  f"{n_cells_total} contributed, "
                  f"λ_max = {lam_max_global:.6e}")

        # Cleanup
        A.destroy()
        B.destroy()

        return lam_max_global

    def store_state(self):
        self.u_old.x.array[:] = self.u.x.array[:]
        self.u_old.x.scatter_forward()

# =============================================================================
# SOLVER
# =============================================================================

class ContactSolver:
    """Solver class for contact problem."""

    def __init__(self, problem: ContactProblem, cfg: SimulationConfig):
        self.problem = problem
        self.cfg = cfg
        self.comm = problem.comm
        self.xi = None
        self.lN_sub_prev = None
        # Measure on contact submesh (created once, reused)
        self.dx_sub = ufl.Measure("dx", domain=problem.contact_mesh)
        # Tracking for augmented-lagrangian metrics (per load step)
        self.last_aug_iters = 0
        # Count of actual load substeps executed (including adaptive cuts)
        self.total_load_steps = 0
        self.total_newton_its = 0.0
        
        base_opts = {
            "snes_type": "newtonls",
            "snes_linesearch_type": cfg.snes_linesearch,
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
            ## debugging Jacobian 
            # "snes_test_jacobian": None,  # check relative error with J_fd
        }

        if cfg.diagnose:
            base_opts.update({
                "snes_linesearch_monitor": None,
                "snes_converged_reason": None,
                "ksp_converged_reason": None,
            })
        self.petsc_options = base_opts
    
    def solve_step(self, d_val, step_idx):
        """
        Solve one load step with selected contact method. 
        
        Parameters
        ----------
        d_val : float
            Current indentation depth (applied as displacement BC)
        step_idx : int
            Current load step index 
            
        Returns
        -------
        status : ConvergenceStatus
        n_iters : int
            Number of Newton iterations at convergence for the current step
        reason : str
            Convergence reason 
        """
        
        if cfg.contact_method == "nitsche":
            return self._solve_standard(d_val, step_idx)
        elif self.cfg.contact_method == "augmented_lagrangian":
            return self._solve_augmented_lagrangian(d_val, step_idx)
        elif self.cfg.contact_method == "uzawa":
            return self._solve_uzawa(d_val, step_idx)
        
        # return self._solve_standard(d_val, step_idx)
    
    def _solve_standard(self, d_val, step_idx):
        """
        Solve contact problem with alternating solution scheme
        between ray-tracing projection and displacement from equilibrium.
        """
        cfg, prob = self.cfg, self.problem
        
        # Alternating parameters
        max_alt_iter = cfg.max_alt_iter if cfg.use_alternating else 1
        if cfg.use_alternating and cfg.xi_relax < 1.0:
            # More iterations needed when damping is active
            max_alt_iter = max(max_alt_iter, int(np.ceil(10 / cfg.xi_relax)))
        alt_tol = 0.01 if cfg.use_alternating else 10
        
        print(f"\n{'='*70}")
        print(f"Starting alternating scheme for d={d_val:.4f}")
        print(f"{'='*70}")
        
        # Store previous displacement and xi for convergence check
        u_prev = prob.u.x.array.copy()
        xi_prev = self.xi.x.array.copy() if self.xi is not None else None
        
        total_newton_its = 0
        
        # ξ relaxation state
        alpha_xi = cfg.xi_relax          # current relaxation factor
        
        # Create NLP once outside the alternating loop
        nlp = NonlinearProblem(
            F=prob.R_total, 
            u=prob.u, 
            bcs=prob.bcs,
            petsc_options=self.petsc_options,
            petsc_options_prefix="indent_c",
            entity_maps=[prob.map_to_parent])
        
        # Quantities on submesh
        self.X_sub = ufl.SpatialCoordinate(prob.contact_mesh)
        self.u_sub = Function(prob.V_sub_)
        try:
            self.u_sub.interpolate_nonmatching(prob.u, cells=np.arange(prob.num_sub_cells),
                                            interpolation_data=prob.interp_data_vec)
        except Exception as e:
            print(f"Warning: interpolate_nonmatching u_sub failed: {e}")
            print("Falling back to zero displacement on submesh for postprocessing.")
            pass

        # Project current normal to DG function on submesh
        self.nx_sub = Function(prob.V_n_sub)
        nx_parent = Function(prob.V_n)
        pts_p = prob.V_n.element.interpolation_points
        pts_p = pts_p() if callable(pts_p) else pts_p
        nx_parent.interpolate(Expression(prob.nx, pts_p))
        self.nx_sub.interpolate_nonmatching(nx_parent, cells=np.arange(prob.num_sub_cells),
                                        interpolation_data=prob.interpolation_data_n)
        
        for alt_iter in range(max_alt_iter):
            # ================================================================
            # Step 1: RAY-TRACING PROJECTION
            # ================================================================
            xi_before_rt = self.xi.x.array.copy() if self.xi is not None else None
            
            xi_rt = ray_tracing_sliding(
                parent_mesh=prob.domain,
                contact_mesh=prob.contact_mesh, 
                facet_marker=prob.facet_tags, 
                c_tag=prob.ft["top"],
                metadata=prob.metadata, 
                nx=prob.nx, 
                x=prob.X + prob.u, 
                xi_guess=self.xi if self.xi is not None else None,
                map_to_parent=prob.map_to_parent, 
                h0=cfg.h0, 
                d_val=d_val, 
                R=cfg.R, 
                y_cap=cfg.y_cap, 
                step_idx=step_idx,
                output_dir=os.path.join(cfg.output_dir, "ray_tracing"), 
                debug_plots=cfg.debug_plots,
                debug_monitor=False,
                u_field=self.u_sub,
                )
            
            # ── Damped ξ update ─────────────────────────────────────
            if xi_before_rt is not None and alpha_xi < 1.0:
                dxi_rt = xi_rt.x.array - xi_before_rt                
                # Blend: ξ_new = (1 − α) ξ_old + α ξ_RT = ξ_old + α (ξ_RT − ξ_old)
                xi_rt.x.array[:] = xi_before_rt + alpha_xi * dxi_rt
                xi_rt.x.scatter_forward()

            self.xi = xi_rt

            # Persist xi back onto the problem placeholder so checkpoints capture it
            try:
                if hasattr(prob, 'xi') and getattr(prob, 'xi', None) is not None and self.xi is not None:
                    prob.xi.x.array[:] = self.xi.x.array[:]
                    try:
                        prob.xi_old.x.array[:] = self.xi.x.array[:]
                    except Exception:
                        pass
                    prob.xi.x.scatter_forward()
                    prob.xi_old.x.scatter_forward()
            except Exception:
                pass
            
            # ================================================================
            # Step 2: UPDATE INDENTER POSITION
            # ================================================================
            # Update positions of projected points on reference plane
            parabola = self.xi**2 / (2*cfg.R)
            y = ufl.as_vector([self.xi, (cfg.h0 - d_val) + ufl.min_value(parabola, cfg.y_cap)])
            
            self.y_sub = Function(prob.V_sub_)
            self.y_sub.interpolate(Expression(y, prob.V_sub_.element.interpolation_points))
            prob.y_func.x.array[:] = self.y_sub.x.array[:]
            prob.y_func.x.scatter_forward()
            
            # ================================================================
            # Step 3: SOLVE MECHANICAL EQUILIBRIUM WITH CONTACT
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
                    reason_name = SNES_REASON_NAMES.get(reason, str(reason))
                    print(f"  Alt iter {alt_iter+1}: SNES failed with reason {reason} ({reason_name})")
                    return ConvergenceStatus.SNES_FAILED, total_newton_its, reason
                
            except Exception as e:
                print(f"    SNES exception: {e}")
                return ConvergenceStatus.EXCEPTION, total_newton_its, None
            
            # ================================================================
            # Step 4: CHECK CONVERGENCE
            # ================================================================
            du = prob.u.x.array - u_prev
            du_norm = np.linalg.norm(du)
            u_norm = np.linalg.norm(prob.u.x.array)
            rel_change_u = du_norm / (u_norm + 1e-10)
            
            if xi_prev is not None:
                dxi = self.xi.x.array - xi_prev
                dxi_norm = np.linalg.norm(dxi)
                xi_norm = np.linalg.norm(self.xi.x.array)
                rel_change_xi = dxi_norm / (xi_norm + 1e-10)
            else:
                V_xi = functionspace(prob.contact_mesh, ("CG", 1))
                X_dofs = V_xi.tabulate_dof_coordinates()[:, 0]
                xi_prev = X_dofs
                dxi_norm = np.linalg.norm(self.xi.x.array - xi_prev)
                xi_norm = np.linalg.norm(self.xi.x.array)
                rel_change_xi = dxi_norm / (xi_norm + 1e-10)

            alpha_str = f", α_ξ={alpha_xi:.3f}" if cfg.use_alternating and cfg.xi_relax < 1.0 else ""
            print(f"  Alt iter {alt_iter+1}: Newton its={n_its}, "
                  f"||du||={du_norm:.4e} (rel={rel_change_u:.4e}), "
                  f"||dξ||={dxi_norm:.4e} (rel={rel_change_xi:.4e}){alpha_str}")

            converged_u = rel_change_u < alt_tol
            converged_xi = rel_change_xi < alt_tol and xi_prev is not None
            
            if converged_u and converged_xi:
                print(f"  -> Alternating minimization converged in {alt_iter+1} iterations")
                print(f"  -> Total Newton iterations: {total_newton_its}")
                return ConvergenceStatus.CONVERGED, total_newton_its, reason
            
            # Update previous state for next iteration
            u_prev[:] = prob.u.x.array
            xi_prev = self.xi.x.array.copy()
        
        # Maximum iterations reached
        print(f"  -> Alternating minimization reached max iterations ({max_alt_iter})")
        return ConvergenceStatus.MAX_ITERATIONS, total_newton_its, None

    def _solve_uzawa(self, d_val, step_idx):
        """Solve using Uzawa's method for contact (submesh-based multiplier updates).

        - Convergence is checked both on the max penetration (∞-norm)
            and on the relative change of the multiplier lN. 
        """
        
        cfg, prob = self.cfg, self.problem
        
        # Collect Newton iterations for each augmented-lagrangian iteration
        newton_its_list = []
        
        # Store previous multiplier for convergence check 
        self.lN_sub_prev_func = Function(prob.V_sub_scalar)
        self.lN_sub_prev_func.x.array[:] = prob.lN_sub.x.array[:]
        self.lN_sub_prev_func.x.scatter_forward()
        
        for aug_iter in range(cfg.aug_lag_max_iter):
            
            status, n_its, reason = self._solve_standard(d_val, step_idx)
            
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
            self.u_sub = Function(prob.V_sub_)
            try:
                self.u_sub.interpolate_nonmatching(prob.u, cells=np.arange(prob.num_sub_cells),
                                             interpolation_data=prob.interp_data_vec)
            except Exception as e:
                print(f"Warning: interpolate_nonmatching u_sub failed: {e}; using fallback zeros")
                self.u_sub.x.array[:] = 0.0
                self.u_sub.x.scatter_forward()
            
            # Build y on submesh 
            parabola = (self.xi)**2 / (2*cfg.R)
            y = ufl.as_vector([self.xi, (cfg.h0 - d_val) + ufl.min_value(parabola, cfg.y_cap)])
            self.y_sub = Function(prob.V_sub_)
            self.y_sub.interpolate(Expression(y, prob.V_sub_.element.interpolation_points))
            
            # Project current normal to DG function on submesh
            self.nx_sub = Function(prob.V_n_sub)
            nx_parent = Function(prob.V_n)
            pts_p = prob.V_n.element.interpolation_points
            pts_p = pts_p() if callable(pts_p) else pts_p
            nx_parent.interpolate(Expression(prob.nx, pts_p))
            self.nx_sub.interpolate_nonmatching(nx_parent, cells=np.arange(prob.num_sub_cells),
                                          interpolation_data=prob.interpolation_data_n)
            
            # Compute gap on submesh at nodes (for multiplier update)
            gN_sub = Function(prob.V_sub_scalar) # same space as lN_sub
            g_expr = gap_rt(self.nx_sub, self.X_sub + self.u_sub, self.y_sub)
            pts_s = prob.V_sub_scalar.element.interpolation_points
            pts_s = pts_s() if callable(pts_s) else pts_s
            try:
                expr_g = Expression(g_expr, pts_s)
                gN_sub.interpolate(expr_g)
            except Exception as e:
                print(f"Warning: Expression(g_expr) compile failed: {e}; setting gN_sub zeros")
                gN_sub.x.array[:] = 0.0
                gN_sub.x.scatter_forward() 

            # Penetration on contact submesh: L2 norm and max penetration (at quadrature points)
            pen_L2, pen_max = self._compute_penetration()

            # Store lN_sub BEFORE the update (for computing the multiplier change)
            self.lN_sub_prev_func.x.array[:] = prob.lN_sub.x.array[:]
            self.lN_sub_prev_func.x.scatter_forward()

            # 1) Compute the multiplier candidate
            lN_candidate = pneg(prob.lN_sub.x.array + prob.eps_contact * gN_sub.x.array)

            # 2) Compute lN_rel using previous multiplier
            lN_diff = Function(prob.V_sub_scalar)
            lN_diff.x.array[:] = lN_candidate - self.lN_sub_prev_func.x.array
            lN_diff.x.scatter_forward()
            lN_change_sq = self.comm.allreduce(
                fem.assemble_scalar(fem.form(lN_diff**2 * self.dx_sub)), MPI.SUM)
            lN_change = np.sqrt(lN_change_sq)
            # L2 norm of candidate multiplier
            lN_cand_func = Function(prob.V_sub_scalar)
            lN_cand_func.x.array[:] = lN_candidate
            lN_cand_func.x.scatter_forward()
            # lN_norm_sq = self.comm.allreduce(
            #     fem.assemble_scalar(fem.form(lN_cand_func**2 * dx_sub)), MPI.SUM)
            lN_norm_sq = self.comm.allreduce(
                fem.assemble_scalar(fem.form(self.lN_sub_prev_func**2 * self.dx_sub)), MPI.SUM)
            lN_norm = np.sqrt(lN_norm_sq) + 1e-12
            
            lN_rel = lN_change / lN_norm

            print(f"  Aug.Lag iter {aug_iter+1}: max penetration = {pen_max:.4e} (tol: {cfg.aug_lag_tol_gap}), "
                  f"lN rel. change = {lN_rel:.4e} (tol: {cfg.aug_lag_tol_lN})")
            print(f"  Effective LM (min): {np.min(lN_candidate):.4e}")

            # 3) Convergence check: both gap and multiplier converged
            if pen_max < cfg.aug_lag_tol_gap and lN_rel < cfg.aug_lag_tol_lN:
                # Accept the classical update and return
                prob.lN_sub.x.array[:] = lN_candidate
                prob.lN_sub.x.scatter_forward()
                # record metrics on solver for external access
                self.last_aug_iters = aug_iter + 1
                self.total_newton_its = float(sum(newton_its_list)) if newton_its_list else 0.0
                return ConvergenceStatus.CONVERGED, aug_iter + 1, None

            # Multiplier update
            prob.lN_sub.x.array[:] = lN_candidate
            prob.lN_sub.x.scatter_forward()

        # Reached maximum augmentation iterations: record metrics
        self.last_aug_iters = cfg.aug_lag_max_iter
        if len(newton_its_list) > 0:
            self.total_newton_its = float(sum(newton_its_list))
        else:
            self.total_newton_its = 0.0
        
        return ConvergenceStatus.MAX_ITERATIONS, cfg.aug_lag_max_iter, None

    def _solve_augmented_lagrangian(self, d_val, step_idx):
        """Solve contact problem with alternating solution strategy
        and augmented Lagrangian method for contact enforcement.
        """
        cfg, prob = self.cfg, self.problem

        # Alternating parameters
        max_alt_iter = cfg.max_alt_iter if cfg.use_alternating else 1
        if cfg.use_alternating and cfg.xi_relax < 1.0:
            # More iterations needed when damping is active
            max_alt_iter = max(max_alt_iter, int(np.ceil(10 / cfg.xi_relax)))
        alt_tol = 0.01 if cfg.use_alternating else 10
        
        print(f"\n{'='*70}")
        print(f"Starting alternating scheme for d={d_val:.4f}")
        print(f"{'='*70}")
        
        # Store previous displacement and xi for convergence check
        u_prev = prob.u.x.array.copy()
        xi_prev = self.xi.x.array.copy() if self.xi is not None else None
        
        total_newton_its = 0

        # ξ relaxation state
        alpha_xi = cfg.xi_relax          # current relaxation factor

        # Create NonlinearProblem once outside the loop.
        nlp = NonlinearProblem(
            F=prob.residual,
            u=[prob.u, prob.lN],
            J=prob.Jac, 
            bcs=prob.bcs,
            petsc_options=self.petsc_options,
            petsc_options_prefix="indent_c",
            entity_maps=[prob.map_to_parent]
        )

        # Quantities on submesh
        self.X_sub = ufl.SpatialCoordinate(prob.contact_mesh)
        self.u_sub = Function(prob.V_sub_)
        try:
            self.u_sub.interpolate_nonmatching(prob.u, cells=np.arange(prob.num_sub_cells),
                                            interpolation_data=prob.interp_data_vec)
        except Exception as e:
            print(f"Warning: interpolate_nonmatching u_sub failed: {e}")
            print("Falling back to zero displacement on submesh for postprocessing.")
            pass

        # Project current normal to DG function on submesh
        self.nx_sub = Function(prob.V_n_sub)
        nx_parent = Function(prob.V_n)
        nx_parent.interpolate(Expression(prob.nx, prob.V_n.element.interpolation_points))
        self.nx_sub.interpolate_nonmatching(nx_parent, cells=np.arange(prob.num_sub_cells),
                                        interpolation_data=prob.interpolation_data_n)

        for alt_iter in range(max_alt_iter):
            # ================================================================
            # Step 1: RAY-TRACING PROJECTION
            # ================================================================
            xi_before_rt = self.xi.x.array.copy() if self.xi is not None else None

            xi_rt = ray_tracing_sliding(
                parent_mesh=prob.domain,
                contact_mesh=prob.contact_mesh,
                facet_marker=prob.facet_tags,
                c_tag=prob.ft['top'],
                metadata=prob.metadata,
                nx=prob.nx,
                x=prob.X + prob.u,
                xi_guess=self.xi if self.xi is not None else None,
                map_to_parent=prob.map_to_parent,
                h0=cfg.h0,
                d_val=d_val,
                R=cfg.R,
                y_cap=cfg.y_cap, 
                step_idx=step_idx,
                output_dir=os.path.join(cfg.output_dir, "ray_tracing"),
                debug_plots=cfg.debug_plots,
                debug_monitor=False,
                u_field=self.u_sub,
            )
            
            # ── Damped ξ update ─────────────────────────────────────
            if xi_before_rt is not None and alpha_xi < 1.0:
                dxi_rt = xi_rt.x.array - xi_before_rt                
                # Blend: ξ_new = (1 − α) ξ_old + α ξ_RT = ξ_old + α (ξ_RT − ξ_old)
                xi_rt.x.array[:] = xi_before_rt + alpha_xi * dxi_rt
                xi_rt.x.scatter_forward()

            self.xi = xi_rt

            # Persist xi back onto the problem placeholder so checkpoints capture it
            try:
                if hasattr(prob, 'xi') and getattr(prob, 'xi', None) is not None and self.xi is not None:
                    prob.xi.x.array[:] = self.xi.x.array[:]
                    try:
                        prob.xi_old.x.array[:] = self.xi.x.array[:]
                    except Exception:
                        pass
                    prob.xi.x.scatter_forward()
                    prob.xi_old.x.scatter_forward()
            except Exception:
                pass
            
            # ================================================================
            # Step 2: UPDATE INDENTER POSITION
            # ================================================================
            # Update positions of projected points on reference plane
            parabola = self.xi**2 / (2*cfg.R)
            y = ufl.as_vector([self.xi, (cfg.h0 - d_val) + ufl.min_value(parabola, cfg.y_cap)])
            
            self.y_sub = Function(prob.V_sub_)
            self.y_sub.interpolate(Expression(y, prob.V_sub_.element.interpolation_points))
            prob.y_func.x.array[:] = self.y_sub.x.array[:]
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

    def solve_step_adaptive(self, d_val, d_prev, step_idx):
        cfg = self.cfg
        backup = self._backup_state()
        if hasattr(self, 'comm') and self.comm.rank == 0:
            print(f"Adaptive step start: step_idx={step_idx}, d_prev={d_prev:.6g}, d_val={d_val:.6g}")

        # ── γ₀ adaptive strategy ─────────────────────────────
        adaptive_strat = cfg.gamma_adaptive and cfg.contact_method == "nitsche"
        # ──────────────────────────────────────────────────
        total_its = 0
        start_time = time.time()
        last_n_its = None
        last_n_cuts = 0
        last_n_sub = 1

        if adaptive_strat:
            # Recompute γ₀ estimate from generalized eigenvalue problem at current def
            old_lam_max = float(self.problem.lam_max.value)
            # lam_max_new = self.problem.estimate_lmbda_max_contact()
            lam_max_new = self.problem.estimate_lmbda_max_contact_local()
            self.problem.lam_max.value = lam_max_new
            print(f"    Re-estimated λ_max: {old_lam_max:.4e} → {lam_max_new:.4e}")

        for n_cuts in range(cfg.max_step_cuts + 1):
            if n_cuts > 0:
                self._restore_state(backup)

            n_sub = 2 ** n_cuts
            dd = (d_val - d_prev) / n_sub
            if self.comm.rank == 0:
                g0_info = f" (γ₀={float(self.problem.gamma0_const.value):.4e})" if adaptive_strat else ""
                print(f"  Trying n_cuts={n_cuts} -> n_sub={n_sub}, dd={dd:.6g}{g0_info}")

            all_ok = True
            for sub_idx in range(n_sub):
                d_sub = d_prev + (sub_idx+1)*dd
                if self.comm.rank == 0:
                    print(f"    Substep {sub_idx+1}/{n_sub}: d_sub={d_sub:.6g}")

                status, n_its, reason = self.solve_step(d_sub, step_idx)
                total_its += int(n_its or 0)
                last_n_its = n_its

                if hasattr(self, 'comm') and self.comm.rank == 0:
                    print(f"      Result: status={status.name} with code={reason}, its={n_its}")

                if status == ConvergenceStatus.CONVERGED:
                    self.problem.store_state()
                    if hasattr(self, 'comm') and self.comm.rank == 0:
                        print("      Stored state for this substep.")
                else:
                    if hasattr(self, 'comm') and self.comm.rank == 0:
                        print(f"      FAILURE on substep {sub_idx+1}; will attempt refinement (n_cuts={n_cuts}).")
                    all_ok = False
                    break

            if all_ok:
                step_converged = True
                last_n_cuts = n_cuts
                last_n_sub = n_sub
                
                elapsed = time.time() - start_time
                self.total_load_steps += last_n_sub
                if self.comm.rank == 0:
                    g0_str = f", γ₀={float(self.problem.gamma0_const.value):.4e}" if adaptive_strat else ""
                    print(f"Adaptive step success: reached d={d_val:.6g} after n_cuts={last_n_cuts}, "
                            f"substeps={last_n_sub}, total_load_steps={self.total_load_steps}, "
                            f"total_its={total_its}, time={elapsed:.3f}s{g0_str}")
                    return True, {'d': d_val}, last_n_its

        self._restore_state(backup)
        if hasattr(self, 'comm') and self.comm.rank == 0:
            fail_msg = f"Adaptive step failed after max cuts={cfg.max_step_cuts}"
            fail_msg += f"; restoring to previous state d_prev={d_prev}"
            print(fail_msg)
        return False, {'d': d_prev}, last_n_its

    def _backup_state(self):
        prob = self.problem
        bk = {'u': prob.u.x.array.copy(), 'u_old': prob.u_old.x.array.copy(),
                'xi': self.xi.x.array.copy() if self.xi else None}
        # Also backup Lagrange multipliers for augmented Lagrangian
        if hasattr(prob, 'lN_sub'):
            bk['lN_sub'] = prob.lN_sub.x.array.copy()
        if hasattr(prob, 'lN_contact'):
            bk['lN_contact'] = prob.lN_contact.x.array.copy()
        return bk
    
    def _restore_state(self, backup):
        prob = self.problem
        prob.u.x.array[:] = backup['u']
        prob.u.x.scatter_forward()
        prob.u_old.x.array[:] = backup['u_old']
        prob.u_old.x.scatter_forward()
        if backup['xi'] is not None and self.xi:
            self.xi.x.array[:] = backup['xi']
            self.xi.x.scatter_forward()
        # Restore Lagrange multipliers
        if 'lN_sub' in backup and hasattr(prob, 'lN_sub'):
            prob.lN_sub.x.array[:] = backup['lN_sub']
            prob.lN_sub.x.scatter_forward()
        if 'lN_contact' in backup and hasattr(prob, 'lN_contact'):
            prob.lN_contact.x.array[:] = backup['lN_contact']
            prob.lN_contact.x.scatter_forward()

    def _compute_penetration(self):
        """Compute penetration metrics on contact boundary.

        The gap is evaluated at quadrature points on the contact submesh,
        consistent with the integral enforcement of the impenetrability
        constraint in the variational formulation.

        Returns
        -------
        pen_L2 : float
            Integral L²(Γc) norm of penetration: √(∫ min(g,0)² ds).
        pen_max : float
            Maximum pointwise penetration at quadrature points (mesh-independent).
        """
        prob = self.problem
        cfg = self.cfg

        # Refresh submesh kinematics from the CURRENT converged state so
        # penetration metrics are consistent across contact methods.
        self.X_sub = ufl.SpatialCoordinate(prob.contact_mesh)

        self.u_sub = Function(prob.V_sub_)
        try:
            self.u_sub.interpolate_nonmatching(
                prob.u,
                cells=np.arange(prob.num_sub_cells),
                interpolation_data=prob.interp_data_vec,
            )
        except Exception:
            self.u_sub.x.array[:] = 0.0
            self.u_sub.x.scatter_forward()

        self.nx_sub = Function(prob.V_n_sub)
        nx_parent = Function(prob.V_n)
        pts_p = prob.V_n.element.interpolation_points
        pts_p = pts_p() if callable(pts_p) else pts_p
        try:
            nx_parent.interpolate(Expression(prob.nx, pts_p))
            self.nx_sub.interpolate_nonmatching(
                nx_parent,
                cells=np.arange(prob.num_sub_cells),
                interpolation_data=prob.interpolation_data_n,
            )
        except Exception:
            self.nx_sub.x.array[:] = 0.0
            self.nx_sub.x.scatter_forward()

        self.y_sub = Function(prob.V_sub_)
        self.y_sub.x.array[:] = prob.y_func.x.array[:]
        self.y_sub.x.scatter_forward()

        g = gap_rt(self.nx_sub, self.X_sub + self.u_sub, self.y_sub)

        # min(g, 0) = penetration (negative part of the gap)
        pen_expr = ufl.min_value(g, 0.0)

        # L2-norm of penetration: sqrt( ∫_Γc min(g,0)² ds )
        pen_L2_sq = self.comm.allreduce(
            fem.assemble_scalar(fem.form(pen_expr**2 * self.dx_sub)), MPI.SUM)
        pen_L2 = np.sqrt(pen_L2_sq)

        # Max penetration (evaluated at quadrature points)
        g_func = Function(prob.V_q) 
        g_func.interpolate(Expression(g, prob.V_q.element.interpolation_points))
        g_func.x.scatter_forward()  
        dof_coords = prob.V_q.tabulate_dof_coordinates()
        self.x_quad_g = dof_coords[:, 0]  # x-coordinates of quadrature points
        self.gap_vals = g_func.x.array
        pen_vals = self.gap_vals[self.gap_vals < 0.0]
        pen_max = np.max(np.abs(pen_vals)) if pen_vals.size > 0 else 0.0

        # Compute actual LM 
        # Compute gap on submesh at nodes (for LM postproc)
        gN_sub = Function(prob.V_lm) # same space as lN
        g_expr = g
        pts_s = prob.V_lm.element.interpolation_points
        pts_s = pts_s() if callable(pts_s) else pts_s
        try:
            expr_g = Expression(g_expr, pts_s)
            gN_sub.interpolate(expr_g)
        except Exception as e:
            print(f"Warning: Expression(g_expr) compile failed: {e}; setting gN_sub zeros")
            gN_sub.x.array[:] = 0.0
            gN_sub.x.scatter_forward()
        
        if cfg.contact_method == "augmented_lagrangian":
            lN_effective = pneg(prob.lN.x.array + prob.eps_contact * gN_sub.x.array)
            print(f"  Effective LM (min): {np.min(lN_effective):.4e}")

        if self.comm.rank == 0:
            print(f"Penetration check: L2-norm={pen_L2:.4e}, max |pen|={pen_max:.4e}")
             
        return pen_L2, pen_max

    def save_full_checkpoint(self, checkpoint_dir, step, d_val):
        """Save complete solver state to disk for restart."""
        prob = self.problem
        cfg = self.cfg
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:04d}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        save_dict = {
            'u': prob.u.x.array,
            'u_old': prob.u_old.x.array,
            'xi': self.xi.x.array if self.xi is not None else np.array([])
        }

        if cfg.contact_method == "uzawa":
            save_dict['lN_sub'] = prob.lN_sub.x.array
            save_dict['lN_sub_prev'] = self.lN_sub_prev if self.lN_sub_prev is not None else np.array([])
        elif cfg.contact_method == "augmented_lagrangian":
            save_dict['lN'] = prob.lN.x.array
        
        np.savez(os.path.join(checkpoint_path, "state.npz"), **save_dict)
        
        metadata = {
            'step': step,
            'd_val': d_val,
        }
        with open(os.path.join(checkpoint_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(os.path.join(checkpoint_dir, "latest.txt"), 'w') as f:
            f.write(f"checkpoint_{step:04d}")
        
        if self.comm.rank == 0:
            print(f"  -> Checkpoint saved: step={step}, d={d_val:.4f}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load solver state from latest checkpoint."""
        prob = self.problem
        cfg = self.cfg
        
        latest_file = os.path.join(checkpoint_dir, "latest.txt")
        if not os.path.exists(latest_file):
            return None
        
        with open(latest_file, 'r') as f:
            checkpoint_name = f.read().strip()
        
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        state_file = os.path.join(checkpoint_path, "state.npz")
        metadata_file = os.path.join(checkpoint_path, "metadata.pkl")
        
        if not os.path.exists(state_file) or not os.path.exists(metadata_file):
            if self.comm.rank == 0:
                print(f"  -> Checkpoint files missing in {checkpoint_path}")
            return None
        
        data = np.load(state_file)
        
        prob.u.x.array[:] = data['u']
        prob.u.x.scatter_forward()
        prob.u_old.x.array[:] = data['u_old']
        prob.u_old.x.scatter_forward()
                
        if cfg.contact_method == "uzawa":
            lN_data = data.get('lN_sub', np.array([]))
            if lN_data.size > 0:
                prob.lN_sub.x.array[:] = lN_data
                prob.lN_sub.x.scatter_forward()
            lN_sub_prev_data = data.get('lN_sub_prev', np.array([]))
            if lN_sub_prev_data.size > 0:
                self.lN_sub_prev = lN_sub_prev_data.copy()
        elif cfg.contact_method == "augmented_lagrangian":
            lN_data = data.get('lN', np.array([]))
            if lN_data.size > 0:
                prob.lN.x.array[:] = lN_data
                prob.lN.x.scatter_forward()
        
        xi_data = data['xi']
        if xi_data.size > 0 and self.xi is not None:
            self.xi.x.array[:] = xi_data
            self.xi.x.scatter_forward()
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        if self.comm.rank == 0:
            print(f"  -> Checkpoint loaded: step={metadata['step']}, d={metadata['d_val']:.4f}")
        
        return metadata

# =============================================================================
# RESULTS
# =============================================================================

@dataclass
class Results:
    """Container for simulation results."""
    phase: List[str] = field(default_factory=list)
    step: List[int] = field(default_factory=list)
    d_val: List[float] = field(default_factory=list)
    Fn: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    p_quad: List[np.ndarray] = field(default_factory=list)
    x_quad_g: List[np.ndarray] = field(default_factory=list)
    g_quad: List[np.ndarray] = field(default_factory=list)
    newton_its: List[int] = field(default_factory=list)
    aug_iters: List[int] = field(default_factory=list)
    avg_newton_its: List[float] = field(default_factory=list)
    gamma0: List[float] = field(default_factory=list)

    def save(self, filename):
        np.savez(filename, **{k: np.array(v) for k, v in self.__dict__.items()})

# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation(cfg: SimulationConfig, restart=False):
    comm = MPI.COMM_WORLD
    os.makedirs(cfg.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    
    if comm.rank == 0:
        print("="*60)
        print("HYPERELASTIC CONTACT: Rigid indenter on hyperelastic substrate")
        print("=" * 60)     
        if restart:
            print(f"RESTART MODE: Looking for checkpoint in {checkpoint_dir}")
            print("="*60)
    
    # Save simulation parameters for postprocessing
    if comm.rank == 0:
        sim_params_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        np.savez(os.path.join(cfg.output_dir, f"simulation_params_P{cfg.u_degree}.npz"), **sim_params_dict)

    # Initialize problem and solver
    problem = ContactProblem(cfg)
    solver = ContactSolver(problem, cfg)
    results = Results()
    checkpoint_mgr = CheckpointManager(problem, cfg.output_dir)

    # If restart requested, try to load latest full checkpoint via solver
    meta = None
    if restart:
        try:
            if hasattr(problem, 'xi') and getattr(problem, 'xi', None) is not None:
                solver.xi = problem.xi
        except Exception:
            pass

        try:
            meta = solver.load_checkpoint(checkpoint_dir)
            if meta is not None and comm.rank == 0:
                print(f"Restart: loaded checkpoint metadata: {meta}")
        except Exception:
            try:
                meta = checkpoint_mgr.load_latest_checkpoint()
                if meta is not None and comm.rank == 0:
                    print(f"Restart (legacy): loaded checkpoint metadata: {meta}")
            except Exception:
                meta = None
        results_file = os.path.join(cfg.output_dir, f"results_{cfg.u_degree}.npz")
        if os.path.exists(results_file):
            try:
                loaded = np.load(results_file, allow_pickle=True)
                for fld in ['phase', 'step', 'd_val', 'Fn', 'force_tangent', 'energy', 'x_quad', 'p_quad', 'pc_error_L2']:
                    if fld in loaded:
                        arr = loaded[fld]
                        if hasattr(arr, 'dtype') and arr.dtype == object:
                            setattr(results, fld, [a for a in arr.tolist()])
                        else:
                            setattr(results, fld, list(arr))
                if comm.rank == 0:
                    print(f"Restart: loaded previous results with {len(results.step)} steps")
            except Exception as e:
                if comm.rank == 0:
                    print(f"Warning: failed to load previous results: {e}")

    # VTX output (just the displacement)
    vtx = VTXWriter(comm, f"{cfg.output_dir}/results_u.bp", [problem.u], engine="BP4")
        
    # Indentation phase
    d_values = np.linspace(0, cfg.d_final, cfg.n_steps_indent + 1)[1:]
    d_prev = 0.0
    global_step, consecutive_failures = 0, 0
    
    current_state = {'d_val': d_prev, 'step': global_step, 'phase': 'indent'}

    def _signal_handler(sig, frame):
        if comm.rank == 0:
            print("\nReceived signal, saving emergency checkpoint...", flush=True)
        try:
            checkpoint_mgr.save_emergency_checkpoint(problem, current_state['step'], reason=f'signal_{sig}')
            results.save(os.path.join(cfg.output_dir, f"results_emergency_step{current_state['step']}.npz"))
            if comm.rank == 0:
                print("  -> Emergency checkpoint saved", flush=True)
        except Exception as e:
            if comm.rank == 0:
                print(f"  -> Warning: emergency checkpoint failed: {e}", flush=True)
        os._exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Determine start index when restarting
    start_idx = 0
    if meta is not None:
        try:
            global_step = int(meta.get('step', 0))
            d_prev = float(meta.get('d_val', d_prev))
            try:
                start_idx = int(np.searchsorted(d_values, d_prev, side='right'))
                start_idx = max(0, min(start_idx, len(d_values)))
                if comm.rank == 0:
                    print(f"Resuming from saved global step {global_step}, mapped d_val={d_prev} to start index {start_idx}")
            except Exception as e:
                start_idx = int(meta.get('step', 0))
                if comm.rank == 0:
                    print(f"Warning: failed to map saved d_val to index: {e}; using saved step {start_idx}")

            current_state.update({'d_val': d_prev, 'step': global_step, 'phase': 'indent'})
        except Exception:
            start_idx = 0


    for i, d_val in enumerate(d_values[start_idx:], start=start_idx):
        if comm.rank == 0:
            print("\n" + "="*40)
            print(f"Indentation step {i+1}/{len(d_values)}: d={d_val:.4f}")
            print("="*40)
        d_prev_before = d_prev
        
        success, info, n_its = solver.solve_step_adaptive(d_val, d_prev, global_step)
        
        # Ensure convergence       
        if success:
            consecutive_failures = 0
            d_prev = d_val

            # Ensure solver exposes augmentation metrics for non-augmented runs (for compatibility)
            if cfg.contact_method != "uzawa":
                solver.last_aug_iters = 0
                # average Newton its is simply the Newton its for single solve
                try:
                    solver.total_newton_its = float(n_its)
                except Exception:
                    solver.total_newton_its = 0.0
            
            # Record results
            # Use total psi directly for energy (works for all models including SVK)
            En = comm.allreduce(fem.assemble_scalar(fem.form(problem.psi * problem.dx)), MPI.SUM)
            # Compute split only if the model provides it (non-zero UFL forms)
            if isinstance(problem.psi_vol, (int, float)) and problem.psi_vol == 0.0:
                En_vol, En_iso = 0.0, En  # no split available; attribute all to isochoric
            else:
                En_vol = comm.allreduce(fem.assemble_scalar(fem.form(problem.psi_vol * problem.dx)), MPI.SUM)
                En_iso = comm.allreduce(fem.assemble_scalar(fem.form(problem.psi_iso * problem.dx)), MPI.SUM)
            
            if cfg.contact_method == "nitsche":
                # Compute normal force from 1st Piola-Kirchhoff stress (yy-component)
                Fn = compute_normal_force(problem.domain, problem.P, problem.ds(problem.ft['top']))
                # Pressure distribution at quadrature points
                # sigma_nn = ufl.dot(ufl.dot(problem.sigma, problem.nx), problem.nx)
                sigma_nn = problem.P_n
                quad_deg = problem.metadata["quadrature_degree"]
                contact_facets = problem.facet_tags.find(problem.ft['top']) 
                x_quad, sigma_nn_quad, _ = expression_at_quadrature(
                    problem.domain, sigma_nn, quad_deg, contact_facets)
                p_quad = -sigma_nn_quad  # Contact pressure 
            elif cfg.contact_method == "uzawa":
                Fn_form = fem.form(-problem.lN_sub * solver.dx_sub)  # LM integrated over contact boundary gives normal force
                Fn = comm.allreduce(fem.assemble_scalar(Fn_form), op=MPI.SUM)   
                # LM at quadrature points on reference configuration
                lN_quad = Function(problem.V_q, name="lN_quad")
                lN_quad.interpolate(Expression(problem.lN_sub, problem.V_q.element.interpolation_points))
                lN_quad.x.scatter_forward()
                p_quad = - lN_quad.x.array 
            elif cfg.contact_method == "augmented_lagrangian":
                Fn_form = fem.form(-problem.lN * solver.dx_sub)
                Fn = comm.allreduce(fem.assemble_scalar(Fn_form), op=MPI.SUM)  
                # LM at quadrature points on reference configuration
                lN_quad = Function(problem.V_q, name="lN_quad")
                lN_quad.interpolate(Expression(problem.lN, problem.V_q.element.interpolation_points))
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

            # # # Nodal positions on contact submesh in current configuration
            # # x_current = fem.Function(problem.V_sub_)
            # # x_current_expr = solver.X_sub + solver.u_sub
            # # x_current.interpolate(fem.Expression(x_current_expr, problem.V_sub_.element.interpolation_points))
            # # x_current.x.scatter_forward()

            # # ### Extract contact pressure from Cauchy at nodes (to match <CPRESS> in Abaqus)
            # # pc_ufl = -ufl.dot(ufl.dot(problem.sigma, problem.nx), problem.nx) 
            # # pc_parent = Function(problem.V_scalar)  # Scalar CG1 on parent mesh
            # # pc_parent.interpolate(Expression(pc_ufl, problem.V_scalar.element.interpolation_points))
            # # pc_parent.x.scatter_forward()
            # # # Interpolate onto submesh 
            # # pc_sub_nodes = Function(problem.V_sub_scalar)
            # # pc_sub_nodes.interpolate_nonmatching(pc_parent, cells=np.arange(problem.num_sub_cells),
            # #                               interpolation_data=problem.interp_data_scalar)

            # Extract gap values (at quadrature points) for monitoring 
            gap_error_L2, max_pen = solver._compute_penetration()
            g_quad, x_quad_g = solver.gap_vals, solver.x_quad_g
                        
            if comm.rank == 0:
                print("--- Monitoring ---")
                print(f"Elastic Energy: Volumetric {En_vol:.3e}, Isochoric {En_iso:.3e}, Total {En:.3e}")
                print(f"Normal contact force Fn: {Fn:.4f} N")
                print(f"Contact pressure (quad points): min: {p_quad.min():.4e}, max: {p_quad.max():.4e}")
                print(f"Gap function (quad points): min: {g_quad.min():.4e}, max: {g_quad.max():.4e}")


            results.phase.append('indent')
            results.step.append(global_step)
            results.d_val.append(d_val)
            results.Fn.append(Fn)
            results.energy.append(En)
            results.p_quad.append(p_quad)
            results.x_quad_g.append(x_quad_g)
            results.g_quad.append(g_quad)
            results.newton_its.append(n_its)
            results.aug_iters.append(solver.last_aug_iters)
            results.avg_newton_its.append(solver.total_newton_its)
            if cfg.contact_method == "nitsche":
                results.gamma0.append(float(problem.gamma0_const.value))

            # Save intermediate results 
            results.save(f"{cfg.output_dir}/results_P{cfg.u_degree}.npz")

            global_step += 1
            current_state.update({'d_val': d_val, 'step': global_step, 'phase': 'indent'})
            
            # Write current displacement to VTX for visualization            
            vtx.write(float(global_step))
            try:
                if global_step % cfg.checkpoint_interval == 0:
                    d_incr = d_val - d_prev_before
                    solver.save_full_checkpoint(checkpoint_dir, global_step, d_val)
            except Exception as e:
                if comm.rank == 0:
                    print(f"Warning: checkpoint save failed: {e}")
        else:
            print("Max refinements reached, aborting")
            break

    vtx.close()
    
    if comm.rank == 0:
        print("\n" + "="*60)
        print(f"Simulation complete! Results in {cfg.output_dir}")
        if not restart:
            print(f"  Planned load steps : {cfg.n_steps_indent}") 
        print(f"  Actual load steps  : {solver.total_load_steps}  (including adaptive cuts)")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rigid parabolic indenter on hyperelastic substrate with various contact enforcement methods (Nitsche, Uzawa, Augmented Lagrangian)."
    )
    
    parser.add_argument("-o", "--output", type=str, default="output_indentation_nitsche",
                        help="Base output directory for results and checkpoints")
    parser.add_argument("-r", "--restart", action="store_true",
                        help="Restart from latest checkpoint in the output directory (if any)")
    parser.add_argument("--material", type=str, default="NH_ss",
                        help="Hyperelastic model from constitutive.py")
    parser.add_argument("--u-degree", type=int,  
                        help="Displacement polynomial degree (default: 1)")
    parser.add_argument("--contact-method", choices=["nitsche", "uzawa", "augmented_lagrangian"], default="nitsche",
                        help="Contact enforcement method") 
    parser.add_argument("--gamma0", type=float, 
                        help="Nitsche penalty parameter γ₀ (default: 1.0)")
    parser.add_argument("--theta", type=int, choices=[0, 1, -1],
                        help="Nitsche parameter for variant selection: 1=symmetric, -1=skew, 0=unsymmetric (default: 0)")
    parser.add_argument("--gamma-adapt", action="store_true",
                        help="Enable γ₀ adaptive strategy (Nitsche only)")
    parser.add_argument("--LcMin", type=float, 
                        help="Minimum mesh size")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Max SNES iterations per solve")
    parser.add_argument("--max-step-cuts", type=int, default=4,
                        help="Max adaptive step bisections")
    parser.add_argument("--diagnose", action="store_true",
                        help="Enable detailed convergence diagnostics")
    parser.add_argument("--snes-linesearch", type=str, default="bt",
                        choices=["bt", "l2", "cp", "nleqerr", "basic"],
                        help="PETSc SNES line search type (dafault: backtracking 'bt')")
    parser.add_argument("--use-altern", action="store_true", 
                        help="Use alternating scheme for ray-tracing-displ solution (default: False)")
    parser.add_argument("--xi-relax", type=float, default=1.0,
                        help="ξ under-relaxation factor α∈(0,1] for alternating scheme (default: 1.0 = no damping)")

    args = parser.parse_args()
    
    cfg = SimulationConfig()
    cfg.output_dir = "output_indentation_" + args.contact_method + "_rt"

    if args.output is not None:
        cfg.output_dir = args.output
    
    if args.contact_method:
        cfg.contact_method = args.contact_method
    
    if args.material is not None:
        cfg.hyperelastic_model = args.material
    
    if args.u_degree is not None:
        cfg.u_degree = args.u_degree
    
    if args.use_altern:
        cfg.use_alternating = True
    
    if args.gamma0 is not None:
        cfg.nitsche_gamma0 = args.gamma0
    
    if args.theta is not None:
        cfg.nitsche_theta = args.theta
    
    if args.gamma_adapt:
        cfg.gamma_adaptive = True

    if args.LcMin is not None:
        cfg.LcMin = args.LcMin
    
    if args.max_iter is not None:
        cfg.max_iter = args.max_iter
    
    if args.max_step_cuts is not None:
        cfg.max_step_cuts = args.max_step_cuts
    
    if args.diagnose:
        cfg.diagnose = True
    
    if args.snes_linesearch is not None:
        cfg.snes_linesearch = args.snes_linesearch
    
    if args.xi_relax is not None:
        cfg.xi_relax = args.xi_relax

        
    # Recompute derived parameters (penalty, tolerances, etc.) after CLI overrides
    cfg.recompute_derived()
    
    if args.restart:
        print("=" * 60)
        print(f"RESTART MODE ENABLED")
        print(f"Output directory: {cfg.output_dir}")
        print("=" * 60)
    
    run_simulation(cfg, restart=args.restart)
