#!/usr/bin/env python3
"""
Constitutive models for hyperelastic materials.
"""

import ufl
from dolfinx import fem


def compute_kinematics(u, ndim, linear=False):
    """
    Compute kinematic quantities from displacement field.
    
    Returns dict with: F, J, C, E, I1, I2, I1_dev, I2_dev
    """
    I = ufl.Identity(ndim)
    H = ufl.grad(u)
    
    if linear:
        # Small strain quantities (approximation)
        eps = 0.5 * (H + H.T)
        F = I  # No deformation gradient in linear theory
        J = 1.0  # No volume change calculation
        C = I
        E = eps
        
        # For compatibility with hyperelastic code
        I1 = ufl.tr(2*eps + I)  # Approximate
        I2 = I1  # Not used in linear
        
        return {
            'I': I, 'F': I, 'J': J, 'C': I, 'E': eps,
            'I1': I1, 'I2': I2, 'I1_bar': I1, 'I2_bar': I2,
            'eps': eps, 'linear': True
        }
    
    else:
        F = ufl.variable(I + H)
    
        J = ufl.det(F)
        C = ufl.variable(F.T * F)
        E = ufl.variable(0.5 * (C - I))
        
        # Invariants (plane-strain)
        F_3D = ufl.as_tensor([[F[0, 0], F[0, 1], 0],
                            [F[1, 0], F[1, 1], 0],
                            [0,         0,     1]])
        C_3D = F_3D.T * F_3D

        I1 = ufl.tr(C_3D)
        I2 = 0.5 * (I1**2 - ufl.inner(C_3D,C_3D))
        # =====================================================================
        # ISOCHORIC INVARIANTS
        # =====================================================================
        I1_bar = J**(-2/3) * I1
        I2_bar = J**(-4/3) * I2
        
        return {
            'I': I, 'F': F, 'J': J, 'C': C, 'E': E,
            'I1': I1, 'I2': I2, 'I1_bar': I1_bar, 'I2_bar': I2_bar
        }

def saint_venant_kirchhoff(domain, kin, Y, nu):
    """
    Saint Venant-Kirchhoff strain energy density.
    
    Uses full Green-Lagrange strain: E = 0.5*(F^T F - I)
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    kin : dict
        Kinematic quantities from compute_kinematics()
    Y : float
        Young's modulus
    nu : float
        Poisson's ratio
        
    Returns
    -------
    psi : ufl form
        Strain energy density
    psi_dev, psi_vol : ufl forms
        Deviatoric and volumetric parts
    params : dict
        Material parameters as Constants
    """
    lam = nu * Y / ((1 + nu) * (1 - 2 * nu))
    mu_ = Y / (2 * (1 + nu))
    k = lam + 2 * mu_ / 3
    
    mu = fem.Constant(domain, mu_)
    lam_const = fem.Constant(domain, lam)
    kappa = fem.Constant(domain, k)
    
    E = kin['E']
    J = kin['J']

    psi = mu * ufl.inner(E, E) + 0.5 * lam_const * (ufl.tr(E))**2
    # psi = kappa/2 * ufl.ln(J)**2 + mu * ufl.inner(E, E) # Modified form from Holzapfel's book
    
    return psi, 0.0, 0.0, {'mu': mu, 'lam': lam_const, 'Y': Y, 'nu': nu}

def linear_elastic(domain, kin, Y, nu):
    """
    Linear elastic strain energy density (small strain).
    
    Uses linearized strain: eps = 0.5*(grad(u) + grad(u)^T)
    
    """
    lam = nu * Y / ((1 + nu) * (1 - 2 * nu))
    mu_ = Y / (2 * (1 + nu))
    
    mu = fem.Constant(domain, mu_)
    lam_const = fem.Constant(domain, lam)
    
    # Use the linearized strain already computed in compute_kinematics
    eps = ufl.variable(kin['E'])
    
    psi = mu * ufl.inner(eps, eps) + 0.5 * lam_const * (ufl.tr(eps))**2
    
    return psi, 0.0, 0.0, {'mu': mu, 'lam': lam_const, 'Y': Y, 'nu': nu, 'eps': eps}

def neo_hookean_ss(domain, kin, Y, nu):
    """
    Neo-Hookean strain energy density version for small strains
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    kin : dict
        Kinematic quantities from compute_kinematics()
    Y : float
        Young's modulus
    nu : float
        Poisson's ratio
        
    Returns
    -------
    psi : ufl form
        Strain energy density
    psi_dev, psi_vol : ufl forms
        Deviatoric and volumetric parts
    params : dict
        Material parameters as Constants
    """
    lam = nu * Y / ((1 + nu) * (1 - 2 * nu))
    mu_ = Y / (2 * (1 + nu))
    k = lam + 2 * mu_ / 3
    
    mu = fem.Constant(domain, mu_)
    kappa = fem.Constant(domain, k)
    
    I1 = kin['I1']
    I1_bar = kin['I1_bar']
    J = kin['J']
    C = kin['C']
    
    psi_dev = 0.5 * mu * (I1 - 3)  
    # psi_vol = lam/4 * (J**2 - 1) - (mu/2 + lam/4) * (ufl.ln(J**2))
    psi_vol = - mu * ufl.ln(J) + lam/2 * (ufl.ln(J))**2  # reduces to linear elasticity at small strains
    psi = psi_dev + psi_vol
    
    return psi, psi_dev, psi_vol, {'mu': mu, 'kappa': kappa, 'Y': Y, 'nu': nu}

def neo_hookean(domain, kin, Y, nu):
    """
    Neo-Hookean strain energy density (Abaqus form).
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    kin : dict
        Kinematic quantities from compute_kinematics()
    Y : float
        Young's modulus
    nu : float
        Poisson's ratio
        
    Returns
    -------
    psi : ufl form
        Strain energy density
    psi_dev, psi_vol : ufl forms
        Deviatoric and volumetric parts
    params : dict
        Material parameters as Constants
    """
    lam = nu * Y / ((1 + nu) * (1 - 2 * nu))
    mu_ = Y / (2 * (1 + nu))
    k = lam + 2 * mu_ / 3
    
    mu = fem.Constant(domain, mu_)
    kappa = fem.Constant(domain, k)
    
    I1 = kin['I1']
    I1_bar = kin['I1_bar']
    J = kin['J']
    
    psi_dev = 0.5 * mu * (I1_bar - 3)
    psi_vol = 0.5 * kappa * (J - 1)**2
    psi = psi_dev + psi_vol
    
    return psi, psi_dev, psi_vol, {'mu': mu, 'kappa': kappa, 'Y': Y, 'nu': nu}

def mooney_rivlin(domain, kin, Y, nu):
    """
    Mooney-Rivlin strain energy density (compressible).
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    kin : dict
        Kinematic quantities from compute_kinematics()
    Y : float
        Young's modulus  
    nu : float
        Poisson's ratio
        
    Returns
    -------
    psi : ufl form
        Strain energy density
    psi_dev, psi_vol : ufl forms
        Deviatoric and volumetric parts
    params : dict
        Material parameters as Constants
    """
    lam = nu * Y / ((1 + nu) * (1 - 2 * nu))
    mu_ = Y / (2 * (1 + nu))
    k = lam + 2 * mu_ / 3
    
    C01 = fem.Constant(domain, mu_ / 6)
    C10 = fem.Constant(domain, mu_ / 3)
    kappa = fem.Constant(domain, k)
    
    I1_dev = kin['I1_bar']
    I2_dev = kin['I2_bar']
    J = kin['J']
    
    psi_dev = C10 * (I1_dev - 3) + C01 * (I2_dev - 3)
    psi_vol = 0.5 * kappa * (J - 1)**2
    psi = psi_dev + psi_vol
    
    return psi, psi_dev, psi_vol, {'C01': C01, 'C10': C10, 'kappa': kappa, 'Y': Y, 'nu': nu}

def get_constitutive_model(model_name, domain, kin, Y, nu):
    """
    Factory function for constitutive models.
    
    Parameters
    ----------
    model_name : str
        One of: 'NH' (Neo-Hookean), 'MR' (Mooney-Rivlin), 'NH_ss' (Neo-Hookean small strain), 
        'SVK' (Saint Venant-Kirchhoff), 'LINEAR' (true linear elastic)
    domain : dolfinx.mesh.Mesh
    kin : dict
        Kinematic quantities
    Y : float
        Young's modulus
    nu : float
        Poisson's ratio
        
    Returns
    -------
    psi, psi_dev, psi_vol, params
    """
    models = {
        'SVK': saint_venant_kirchhoff,
        'NH_ss': neo_hookean_ss,
        'NH': neo_hookean,
        'MR': mooney_rivlin,
        'LINEAR': linear_elastic,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](domain, kin, Y, nu)

def compute_stress(psi, F, J):
    """
    Compute stress measures from strain energy density.
    
    Returns
    -------
    P : First Piola-Kirchhoff stress
    S : Second Piola-Kirchhoff stress
    sigma : Cauchy stress
    """
    P = ufl.diff(psi, F)
    S = ufl.dot(ufl.inv(F), P)
    sigma = P * F.T / J

    return P, S, sigma

def compute_stress_linear(psi, eps):
    """
    Compute Cauchy stress for linear elastic (small strain) formulation.
    
    For small strains, σ = P = S (all stress measures coincide).
    
    Parameters
    ----------
    psi : ufl form
        Strain energy density ψ(ε)
    eps : ufl variable
        Linearized strain tensor ε = ½(∇u + ∇u^T)
    
    Returns
    -------
    sigma : Cauchy stress σ = ∂ψ/∂ε
    """
    sigma = ufl.diff(psi, eps)  # eps must already be a ufl.variable
    # For linear elasticity, P ≈ S ≈ σ
    return sigma, sigma, sigma

