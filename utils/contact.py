#!/usr/bin/env python3
"""
Contact mechanics utilities. 
"""

import numpy as np
import basix
import ufl
from dolfinx import fem, mesh as dolfinx_mesh
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx


def gap_rt(nx, x, y):
    """
    Gap function for ray-tracing mapping.
    
    Parameters
    ----------
    nx : ufl vector
        Outward normal on slave body in current config
    x : ufl vector
        Current position of slave body (X + u_x)
    y : ufl vector
        Current position of master body (Y + u_y)
        
    Returns
    -------
    g : ufl scalar
        normal Gap (positive = separation, negative = penetration)
    """
    return ufl.dot(nx, (y - x))

def gap_proj(ny, x, y):
    """
    Gap function for projection mapping.
    
    Parameters
    ----------
    ny : ufl vector
        Outward normal on master body in current config
    x : ufl vector
        Current position of slave body (X + ux)
    y : ufl vector
        Current position of master body (Y + uy)
        
    Returns
    -------
    g : ufl scalar
        normal Gap (positive = separation, negative = penetration)
    """
    return ufl.dot(ny, (x - y))

def pneg(x, eps =1e-8):
    """Macaulay bracket: returns min(x, 0)"""
    return 0.5 * (x - abs(x))
    # return 0.5 * (x - ufl.sqrt(x**2 + eps**2))  # smooth approximation of min(x,0) 

def create_contact_submesh(domain, facet_tags, contact_tag, element_deg, ndim):
    """
    Create submesh along contact boundary.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        parent mesh
    facet_tags : dolfinx.mesh.MeshTags
        facet markers for the parent mesh
    contact_tag : int
        marker value for the contact boundary facets
    element_deg : int
        polynomial degree for function spaces on contact submesh
    ndim : int
        spatial dimension of the problem

    Returns
    -------
    contact_mesh : dolfinx.mesh.Mesh
        submesh on contact boundary
    map_to_parent : array
        Entity map from submesh to parent
    V_sub : fem.FunctionSpace
        Vector function space on submesh
    V_parent : fem.FunctionSpace
        Vector function space on parent mesh
    interpolation_data : InterpolationData
        For interpolating from parent to submesh
    """
    fdim = domain.topology.dim - 1
    contact_facets = facet_tags.find(contact_tag)
    
    contact_mesh, map_to_parent, _, _ = dolfinx_mesh.create_submesh(
        domain, fdim, contact_facets
    )
    
    sub_cell_map = contact_mesh.topology.index_map(contact_mesh.topology.dim)
    num_sub_cells = sub_cell_map.size_local + sub_cell_map.num_ghosts
    
    # Function space on submesh
    V_sub = fem.functionspace(contact_mesh, ("CG", element_deg, (ndim,)))
    
    # For parent -> submesh interpolation
    V_parent = fem.functionspace(domain, ("CG", element_deg, (ndim,)))
    interpolation_data = fem.create_interpolation_data(
        V_sub, V_parent, cells=np.arange(num_sub_cells)
    )
    
    return contact_mesh, map_to_parent, V_sub, V_parent, interpolation_data, num_sub_cells

def project_vector_on_boundary(vector_expr, V, facet_tags, facet_id, metadata=None):
    """
    Project any vector expression onto a function space over a boundary.
    Adapted from the GitHub Gist
    "Facet normal and facet tangent vector approximation in dolfinx"
    by Halvor Herlyng.
    
    Original file header:
        Copyright (C) 2023 Jørgen S. Dokken and Halvor Herlyng
        SPDX-License-Identifier: MIT
        
    Parameters
    ----------
    vector_expr : ufl expression
        The vector expression to project (e.g., deformed normal)
    V : fem.FunctionSpace
        Vector function space to project into (typically DG)
    facet_tags : MeshTags
        Facet markers
    facet_id : int
        The facet marker value for the boundary of interest
    metadata : dict, optional
        Quadrature metadata
        
    Returns
    -------
    n_proj : fem.Function
        Projected and normalized vector field
    """    
    metadata = metadata or {"quadrature_degree": 4}
    
    # Define variational problem for L2 projection on boundary
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    ds = ufl.Measure("ds", domain=V.mesh, subdomain_data=facet_tags, 
                     subdomain_id=facet_id, metadata=metadata)
    
    # L2 projection: find u in V such that (u, v)_Γ = (vector_expr, v)_Γ
    a = ufl.inner(u, v) * ds  # Mass matrix on boundary
    L = ufl.inner(vector_expr, v) * ds  # RHS: project your vector expression
    
    # Identify DOFs on the boundary
    ones = fem.Constant(V.mesh, dolfinx.default_scalar_type((1,) * V.mesh.geometry.dim))
    local_form = fem.form(ufl.dot(ones, v) * ds)
    local_vec = fem.assemble_vector(local_form)
    
    # DOFs with zero value are NOT on this boundary
    inactive_dofs = np.flatnonzero(np.isclose(local_vec.array, 0))
    inactive_blocks = np.unique(inactive_dofs // V.dofmap.bs).astype(np.int32)
    
    # Create BC to deactivate interior DOFs
    u_zero = fem.Function(V)
    # dolfinx 0.11+: use the Function.x PETSc vector / array API instead of `.vector`
    u_zero.x.array[:] = 0.0
    u_zero.x.scatter_forward()
    bc_inactive = fem.dirichletbc(u_zero, inactive_blocks)
    
    # Assemble matrix with special handling for inactive DOFs
    bilinear_form = fem.form(a)
    pattern = fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(inactive_blocks)
    pattern.finalize()
    
    A = dolfinx.cpp.la.petsc.create_matrix(V.mesh.comm, pattern)
    A.zeroEntries()
    
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(bilinear_form._cpp_object)
    form_consts = dolfinx.cpp.fem.pack_constants(bilinear_form._cpp_object)
    fem.petsc.assemble_matrix(A, bilinear_form, constants=form_consts, 
                              coeffs=form_coeffs, bcs=[bc_inactive])
    
    # Insert identity on diagonal for inactive DOFs
    A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
    # insert_diagonal expects C++ FunctionSpace and C++ DirichletBC objects
    dolfinx.cpp.fem.petsc.insert_diagonal(A=A, V=V._cpp_object,
                                          bcs=[bc_inactive._cpp_object], diagonal=1.0)
    A.assemble()
    
    # Assemble RHS
    linear_form = fem.form(L)
    b = fem.petsc.assemble_vector(linear_form)
    fem.petsc.apply_lifting(b, [bilinear_form], [[bc_inactive]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc_inactive])
    
    # Solve
    solver = PETSc.KSP().create(V.mesh.comm)
    solver.setType("cg")
    solver.setTolerances(rtol=1e-8)
    solver.setOperators(A)
    
    vec_proj = fem.Function(V)
    # Solve into PETSc vector backing the Function (dolfinx 0.11+)
    solver.solve(b, vec_proj.x.petsc_vec)
    # Update ghost values / local array
    vec_proj.x.scatter_forward()
    
    return vec_proj

def compute_normal_force(domain, P, ds_top):
        """
        Compute total normal force on contact surface.
        
        Uses First Piola-Kirchhoff stress P integrated over reference config.
        This equals the physical force since: ∫_Γ t dA = ∫_Γ₀ P·N dA₀
        
        For top surface with N = (0,1), the normal force is P[·,1] component.
        """
        # P·N gives traction in reference config, P[i,1] is y-component of P·(0,1)
        Fn_form = fem.form(-P[1, 1] * ds_top)
        Fn = fem.assemble_scalar(Fn_form)
        return domain.comm.allreduce(Fn, op=MPI.SUM)

def extract_boundary_cells(domain, facet_tags, c_tag):
        """Get cells adjacent to contact boundary."""
        tagged_facets = facet_tags.indices[facet_tags.values == c_tag]
        
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        facet_to_cell = domain.topology.connectivity(domain.topology.dim - 1, domain.topology.dim)
        
        adjacent_cells = set()
        for f in tagged_facets:
            adjacent_cells.update(facet_to_cell.links(f))
        
        return np.array(sorted(adjacent_cells), dtype=np.int32), tagged_facets, facet_to_cell

def expression_at_quadrature(domain, ufl_expr, deg_quad, contact_facets):
    """
    Evaluate UFL expression at quadrature points on contact facets.
    
    Returns
    -------
    x_sorted : array
        x-coordinates of quadrature points (sorted)
    expr_sorted : array
        Expression values at quadrature points (sorted by x)
    """
    facet_cell = basix.CellType.interval
    points_ref, _ = basix.make_quadrature(facet_cell, deg_quad)
    
    # Map to physical coordinates
    points_phys_expr = fem.Expression(ufl.SpatialCoordinate(domain), points_ref)
    expr = fem.Expression(ufl_expr, points_ref)
    
    # Connectivity
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    f_to_c = domain.topology.connectivity(domain.topology.dim - 1, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    c_to_f = domain.topology.connectivity(domain.topology.dim, domain.topology.dim - 1)
    
    # Build cell-facet pairs
    cell_facet_pairs = np.empty((len(contact_facets), 2), dtype=np.int32)
    for i, facet in enumerate(contact_facets):
        cells = f_to_c.links(facet)
        facets = c_to_f.links(cells[0])
        facet_index = np.flatnonzero(facets == facet)[0]
        cell_facet_pairs[i] = (cells[0], facet_index)
    
    # Evaluate
    coordinates = points_phys_expr.eval(domain, cell_facet_pairs)  # evaluate reference->physical points for each (cell,facet) at quadrature points
    expr_values = expr.eval(domain, cell_facet_pairs)  # evaluate the UFL expression at the same (cell,facet) quadrature points

    # Flatten and sort by x for full quadrature sampling
    x_coords = coordinates[..., 0].reshape(-1)  # all qp x-coordinates flattened across all pairs
    expr_flat = expr_values.reshape(-1)  # all expression values flattened across all pairs/qps
    sort_idx = np.argsort(x_coords)  # sorting indices for these flattened coordinates
    
    return x_coords[sort_idx], expr_flat[sort_idx], expr_values  # return sorted qp x positions and corresponding expr values
        
    
        
        



