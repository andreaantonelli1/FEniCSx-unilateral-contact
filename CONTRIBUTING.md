# How to contribute

Thank you for your interest in this project! 

Contributions of all kinds are welcome, whether they take the form of bug reports, suggestions, or code improvements.

## Scope of the project

This repository provides reference FEniCSx implementations for frictionless unilateral contact in
finite deformations, focusing on 2D contact between a hyperelastic body and a rigid
obstacle. Contact detection is handled via ray-tracing, and enforcement is based on
Nitsche's method and Augmented Lagrangian formulations (mixed and Uzawa).

Possible directions for future developments include extending the framework to 3D,
adapting the codes for parallel execution, incorporating frictional contact,
enhancing and broadening the set of contact enforcement strategies, introducing
multiphysics couplings in the deformable body (viscoelasticity, phase-field fracture ecc..), and expanding the benchmark suite to cover a wider range of configurations.

## Reporting bugs and suggesting enhancements

If you find a bug or want to suggest an improvement, please open an issue on the
[GitHub issue tracker](https://github.com/andreaantonelli1/FEniCSx-unilateral-contact/issues).

For **bug reports**, a useful report includes the following:

- A clear description of the unexpected behavior and what you expected to happen instead.
- The command(s) used to run the script and the full error output or traceback.
- Environment configuration (especially DOLFINx version).

For **enhancement suggestions**, please describe the mathematical or algorithmic
motivation behind your proposal. If your suggestion relates to a known limitation listed
below, feel free to add a comment to the corresponding issue rather than opening a
new one.

## Known limitations

The following are just a few known bugs and planned enhancements:
- Instability in ray-tracing contact detection on unstructured triangular mesh
- Parallel execution via MPI
- Extension to 3D contact