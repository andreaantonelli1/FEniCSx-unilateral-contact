# Unilateral Contact in FEniCSx
This repository collects FEniCSx codes developed during my thesis work on frictionless unilateral contact in finite deformations.

The repository focuses on 2D contact problems between a deformable hyperelastic body and a rigid obstacle, with particular attention to:
- large-deformation kinematics;
- contact detection through ray-tracing;
- different contact enforcement strategies, including Nitsche's method and Augmented Lagrangian formulations (implemented both as mixed formulation and through an Uzawa augmentation loop);
- benchmark problems for validation and comparison.

## Installation

To run the codes, DOLFINx (v0.10.0) is required. It can be installed with [conda](https://anaconda.org/anaconda/conda), see the [download page of FEniCSx](https://fenicsproject.org/download/).
If you prefer working with Docker, a Dockerfile is provided.
To build the image from the root of the repository, just run:

```bash
docker build -t fenicsx_contact .
```

Then, run the container by mounting the current repository inside the container workspace:

```bash
docker run --rm -it -v "$(pwd):/home/user/shared" -w /home/user/shared fenicsx_contact
```

## Usage
The main benchmark scripts are located in `examples/`, while `utils/` contains helper functions, including mesh generation, constitutive models and contact utilities.

For example, to run the Signorini benchmark:

```bash
python3 examples/run_signorini.py
```

You can inspect all command-line options with:

```bash
python3 examples/run_signorini.py --help
```

## Notes on the current state of the codes
The repository is actively developed. The aim of publishing the codes is to make the implementations available, encourage reuse, and possibly receive suggestions for improvements.
See [CONTRIBUTING.md](CONTRIBUTING.md) for a full overview and a list of known
limitations.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
information on the scope of the project, how to report bugs, and known limitations.

## Contact
If you find bugs, have suggestions, or would like to discuss the code, please open an issue in the repository.

For direct contact: [Andrea Antonelli](mailto:antonelli.1958441@studenti.uniroma1.it).
