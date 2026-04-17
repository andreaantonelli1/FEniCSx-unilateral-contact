#!/usr/bin/env python3
import gmsh
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.io.gmsh import model_to_mesh
import typing
import warnings
from pathlib import Path


def elastic_block_sym_DI(ind_center, Dmin, Dmax, ft, ct, h_min, Lx=1.0, Ly=1.0, refinement_ratio = 40, gdim=2, verbosity=1):
    """
    Symmetric half-space -- mesh for (deep) indentation contact.
    
    Domain: [0, 0] x [Lx, -Ly]
    Refinement around the indenter center point.
    
    Parameters
    ----------
    ind_center : tuple (x, y)
        Indenter center position for mesh refinement
    Dmin, Dmax : float
        Distance thresholds for mesh refinement (min and max distance from center)
    ft, ct : dict
        Facet and cell tags
    h_min : float
        Minimum element size near indenter
    Lx, Ly : float
        Domain dimensions (width = Lx, height = Ly)
    refinement_ratio : float
        Ratio between max and min element sizes
    gdim : int
        Geometric dimension (default 2)
    verbosity : int
        GMSH verbosity level

    Returns
    -------
    msh : dolfinx.mesh.Mesh
        The generated mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell tags
    facet_tags : dolfinx.mesh.MeshTags
        Facet tags
    """
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', verbosity)

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    facet_tags = ft
    cell_tags = ct
    
    model = gmsh.model()
    model.add('elastic_block')
    model.set_current('elastic_block')

    xc, yc = ind_center

    # Points 
    p1 = model.geo.addPoint(0,0,0)
    p2 = model.geo.addPoint(0,-Ly,0)
    p3 = model.geo.addPoint(Lx,-Ly,0)
    p4 = model.geo.addPoint(Lx,0,0) 
    pt1 = model.geo.addPoint(xc,yc,0)
    
    # Lines
    l1 = model.geo.addLine(p1, p2, tag=facet_tags["left"])
    l2 = model.geo.addLine(p2, p3, tag=facet_tags["bottom"])
    l3 = model.geo.addLine(p3, p4, tag=facet_tags["right"])
    l4 = model.geo.addLine(p4, p1, tag=facet_tags["top"])
    # Surface
    cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4])
    surf1 = 1 
    surface_1 = model.geo.addPlaneSurface([cloop1], tag=surf1)
    model.geo.synchronize()

    # # Set Physical groups
    # # Surface
    model.addPhysicalGroup(2, [surf1], tag=cell_tags["all"])
    for key, value in facet_tags.items():
        model.addPhysicalGroup(1, [value], tag=value)
        model.setPhysicalName(1, value, key)
    
    # Mesh refinement
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "NodesList", [pt1])
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "IField", 1)
    model.mesh.field.setNumber(2, "LcMin", h_min)
    model.mesh.field.setNumber(2, "LcMax", h_min * refinement_ratio)
    model.mesh.field.setNumber(2, "DistMin", Dmin)
    model.mesh.field.setNumber(2, "DistMax", Dmax)
    model.mesh.field.setAsBackgroundMesh(2)

    model.mesh.generate(gdim)
    
    # Mesh visualization
    #gmsh.fltk.run()

    # # Convert to DOLFINx mesh
    if float(dolfinx.__version__[2:4]) >= 9.0:
        meshdata = model_to_mesh(
            model, mesh_comm, model_rank, gdim=gdim
        )
        msh = meshdata.mesh
        cell_tags = meshdata.cell_tags
        facet_tags = meshdata.facet_tags
    else:
        msh, cell_tags, facet_tags = model_to_mesh(
            model, mesh_comm, model_rank, gdim=gdim
        )
        
    msh.name = "indented_block"
    cell_tags.name = f"{msh.name}_cells"
    facet_tags.name = f"{msh.name}_facets"

    gmsh.finalize()  
    return msh, cell_tags, facet_tags

def elastic_block_DI(ind_center, Dmin, Dmax, ft, ct, h_min, Lx=1.0, Ly=1.0, refinement_ratio = 40, gdim=2, verbosity=1):
    """
    Mesh for half-space in (deep) indentation contact.
    
    Domain: [-Lx/2, 0] x [Lx/2, -Ly]
    Refinement around the indenter center point.
    
    Parameters
    ----------
    ind_center : tuple (x, y)
        Indenter center position for mesh refinement
    Dmin, Dmax : float
        Distance thresholds for mesh refinement (min and max distance from center)
    ft, ct : dict
        Facet and cell tags
    h_min : float
        Minimum element size near indenter
    Lx, Ly : float
        Domain dimensions (width = Lx, height = Ly)
    refinement_ratio : float
        Ratio between max and min element sizes
    gdim : int
        Geometric dimension (default 2)
    verbosity : int
        GMSH verbosity level

    Returns
    -------
    msh : dolfinx.mesh.Mesh
        The generated mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell tags
    facet_tags : dolfinx.mesh.MeshTags
        Facet tags
    """
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', verbosity)

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    facet_tags = ft
    cell_tags = ct
    
    model = gmsh.model()
    model.add('elastic_block')
    model.set_current('elastic_block')

    xc, yc = ind_center

    # Points 
    p1 = model.geo.addPoint(-Lx/2,0,0)
    p2 = model.geo.addPoint(-Lx/2,-Ly,0)
    p3 = model.geo.addPoint(Lx/2,-Ly,0)
    p4 = model.geo.addPoint(Lx/2,0,0) 
    pt1 = model.geo.addPoint(xc,yc,0)
    
    # Lines
    l1 = model.geo.addLine(p1, p2, tag=facet_tags["left"])
    l2 = model.geo.addLine(p2, p3, tag=facet_tags["bottom"])
    l3 = model.geo.addLine(p3, p4, tag=facet_tags["right"])
    l4 = model.geo.addLine(p4, p1, tag=facet_tags["top"])
    # Surface
    cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4])
    surf1 = 1 
    surface_1 = model.geo.addPlaneSurface([cloop1], tag=surf1)
    model.geo.synchronize()

    # # Set Physical groups
    # # Surface
    model.addPhysicalGroup(2, [surf1], tag=cell_tags["all"])
    for key, value in facet_tags.items():
        model.addPhysicalGroup(1, [value], tag=value)
        model.setPhysicalName(1, value, key)
    
    # Mesh refinement
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "NodesList", [pt1])
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "IField", 1)
    model.mesh.field.setNumber(2, "LcMin", h_min)
    model.mesh.field.setNumber(2, "LcMax", h_min * refinement_ratio)
    model.mesh.field.setNumber(2, "DistMin", Dmin)
    model.mesh.field.setNumber(2, "DistMax", Dmax)
    model.mesh.field.setAsBackgroundMesh(2)

    model.mesh.generate(gdim)
    
    # Mesh visualization
    #gmsh.fltk.run()

    # # Convert to DOLFINx mesh
    if float(dolfinx.__version__[2:4]) >= 9.0:
        meshdata = model_to_mesh(
            model, mesh_comm, model_rank, gdim=gdim
        )
        msh = meshdata.mesh
        cell_tags = meshdata.cell_tags
        facet_tags = meshdata.facet_tags
    else:
        msh, cell_tags, facet_tags = model_to_mesh(
            model, mesh_comm, model_rank, gdim=gdim
        )
        
    msh.name = "indented_block"
    cell_tags.name = f"{msh.name}_cells"
    facet_tags.name = f"{msh.name}_facets"

    gmsh.finalize()  
    return msh, cell_tags, facet_tags

def half_disk_mesh(R, LcMin=0.01, LcMax=0.1, Dmin=0.5, Dmax=0.6, filename: typing.Union[str, Path] = "disk.msh"):
    """Create 1/2 disk mesh centered at the origin with radius R.

    Mesh is finer at (0,0) using LcMin, and gradually decreasing to LcMax
    """
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    if MPI.COMM_WORLD.rank == 0:
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        gmsh.model.occ.addPoint(0, 0, 0, tag=1)
        gmsh.model.occ.addPoint(R, R, 0, tag=2)
        gmsh.model.occ.addPoint(-R, R, 0, tag=3)
        gmsh.model.occ.addPoint(0, R, 0, tag=4)

        
        gmsh.model.occ.addLine(3, 2, tag=1)
        gmsh.model.occ.addCircleArc(2, 4, 3, center=True, tag=2)
        
        gmsh.model.occ.addCurveLoop([1, 2], tag=10)
        gmsh.model.occ.addPlaneSurface([10], tag=100)

        
        gmsh.model.occ.synchronize()
        domains = gmsh.model.getEntities(dim=2)
        domain_marker = 11
        gmsh.model.addPhysicalGroup(domains[0][0], [domains[0][1]], domain_marker)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [1])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", LcMin)
        gmsh.model.mesh.field.setNumber(2, "LcMax", LcMax)
        gmsh.model.mesh.field.setNumber(2, "DistMin", Dmin)
        gmsh.model.mesh.field.setNumber(2, "DistMax", Dmax)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)

        gmsh.write(str(filename))

    gmsh.finalize()

    # return msh, cell_tags, facet_tags

def half_disk_mesh_sym(R, a, ft, ct, LcMin=0.01, LcMax=0.1, Dmin=0.5, Dmax=0.6, verbosity=1):
    """Create 1/2 disk mesh centered at the origin with radius R.

    Mesh is finer at (0,0) using LcMin, and gradually decreasing to LcMax
    """
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', verbosity)
    
    if MPI.COMM_WORLD.rank == 0:
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        model = gmsh.model()
        model.add('half_disk_sym')
        model.set_current('half_disk_sym')

        facet_tags = ft
        cell_tags = ct

        # theta_c = np.arcsin(2 * a / R)
        theta_c = np.arctan2(Dmax, R)  
        
        if theta_c > np.pi / 2:
            warnings.warn("Contact half-width exceeds disk radius. Using full half-disk mesh.")
            theta_c = np.pi / 2
        
        x_cont = R * np.sin(theta_c)
        y_cont = R - R * np.cos(theta_c) 

        p1 = model.occ.addPoint(0, 0, 0, tag=1)
        p2 = model.occ.addPoint(R, R, 0, tag=2)
        p3 = model.occ.addPoint(0, R, 0, tag=3)
        pc = model.occ.addPoint(x_cont, y_cont, 0, tag=4)
        # gmsh.model.occ.addPoint(0, R, 0, tag=4)

        
        l1 = model.occ.addLine(p3, p2, tag=facet_tags["top"])
        a1 = model.occ.addCircleArc(p2, p3, pc, tag=facet_tags["free"])
        a2 = model.occ.addCircleArc(pc, p3, p1, tag=facet_tags["contact"])
        l2 = model.occ.addLine(p1, p3, tag=facet_tags["symmetry"])
        
        model.occ.addCurveLoop([l1, a1, a2, l2], tag=10)
        surf_tag = 100
        model.occ.addPlaneSurface([10], tag=surf_tag)
        
        model.occ.synchronize()
        
        # # Set Physical groups
        # # Surface
        model.addPhysicalGroup(2, [surf_tag], tag=cell_tags["all"])
        for key, value in facet_tags.items():
            model.addPhysicalGroup(1, [value], tag=value)
            model.setPhysicalName(1, value, key)
        model.occ.synchronize()
        
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [1])

        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", LcMin)
        model.mesh.field.setNumber(2, "LcMax", LcMax)
        model.mesh.field.setNumber(2, "DistMin", Dmin)
        model.mesh.field.setNumber(2, "DistMax", Dmax)
        model.mesh.field.setAsBackgroundMesh(2)

        model.mesh.generate(2)

    #     gmsh.write(str(filename))

    # gmsh.finalize()
        # # Convert to DOLFINx mesh
        if float(dolfinx.__version__[2:4]) >= 9.0:
            meshdata = model_to_mesh(
                model, mesh_comm, model_rank, gdim=2
            )
            msh = meshdata.mesh
            cell_tags = meshdata.cell_tags
            facet_tags = meshdata.facet_tags
        else:
            msh, cell_tags, facet_tags = model_to_mesh(
                model, mesh_comm, model_rank, gdim=2
            )
            
        msh.name = "half_disk_sym"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"

        gmsh.finalize()  
        return msh, cell_tags, facet_tags

    # return msh, cell_tags, facet_tags

def convert_mesh_new(filename: Path, outname: Path, gdim: int):
    """Read a GMSH mesh (msh format) and convert it to XDMF with both cell
    and facet tags in a single file.

    Args:
        filename:
            Name of input file
        outname:
            Name of output file
        gdim:
            The geometrical dimension of the mesh
    """
    if MPI.COMM_WORLD.rank == 0:
        meshdata = dolfinx.io.gmsh.read_from_msh(filename, MPI.COMM_SELF, 0, gdim=gdim)
        mesh = meshdata.mesh
        ct = meshdata.cell_tags
        ft = meshdata.facet_tags
        with dolfinx.io.XDMFFile(mesh.comm, outname, "w") as xdmf:
            xdmf.write_mesh(mesh)
            # Write cell tags if they exist
            if ct is not None:
                xdmf.write_meshtags(ct, mesh.geometry)
            # Write facet tags if they exist
            if ft is not None:
                mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
                xdmf.write_meshtags(ft, mesh.geometry)
    MPI.COMM_WORLD.Barrier()



    