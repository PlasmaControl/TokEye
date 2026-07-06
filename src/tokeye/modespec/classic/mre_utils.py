"""
mre_utils.py — Modified Rutherford Equation (MRE) helper functions.

References:
  La Haye et al., Phys. Plasmas 9, 2051 (2002)
  Sauter et al., Phys. Plasmas 4, 1654 (1997)
  Hegna & Callen, Phys. Plasmas 4, 2940 (1997)

All quantities SI unless noted.
"""

import numpy as np

MU0 = 4 * np.pi * 1e-7   # H/m


def resistive_time(eta, r_s, prefactor=1.22):
    """
    Resistive diffusion time at the rational surface.

    τ_R = μ₀ r_s² / (prefactor * η)

    Parameters
    ----------
    eta       : float  resistivity [Ω·m]
    r_s       : float  minor radius of q=2 surface [m]
    prefactor : float  numerical factor (1.22 from resistive MHD; sometimes 1.0)

    Returns
    -------
    tau_R : float  [s]
    """
    return MU0 * r_s**2 / (prefactor * eta)


def mre_rhs(W, delta_prime, delta_bs=0.0, delta_eccd=0.0, delta_pol=0.0):
    """
    RHS of the Modified Rutherford Equation (normalized).

    (τ_R / r_s) * dW/dt = Δ' + Δ'_bs + Δ'_ECCD + Δ'_pol

    Parameters
    ----------
    W           : float or array  island half-width [m]
    delta_prime : float           classical tearing index Δ' [m⁻¹]
    delta_bs    : float           bootstrap current term [m⁻¹]  (positive = destabilizing)
    delta_eccd  : float           ECCD term [m⁻¹]  (negative = stabilizing)
    delta_pol   : float           polarization current term [m⁻¹]

    Returns
    -------
    float or array : (τ_R / r_s) * dW/dt  [m⁻¹]
    """
    return delta_prime + delta_bs + delta_eccd + delta_pol


def delta_prime_bootstrap(W, beta_pol, r_s, L_q, L_p, rho_i, C_bs=1.0):
    """
    Bootstrap current contribution to the MRE (simplified NTM form).

    Δ'_bs ≈ C_bs * (β_pol / r_s) * (r_s / L_q) * (r_s / L_p)
              * W / (W² + ρ_i²)

    The ρ_i² term in the denominator suppresses the bootstrap drive below
    the ion orbit width (ion polarization threshold).

    Parameters
    ----------
    W        : float or array  island half-width [m]
    beta_pol : float           poloidal beta
    r_s      : float           rational surface minor radius [m]
    L_q      : float           q-profile scale length: L_q = q / (dq/dr) [m]
    L_p      : float           pressure scale length: L_p = -p / (dp/dr) [m]
    rho_i    : float           ion poloidal Larmor radius [m]
    C_bs     : float           order-unity geometry coefficient

    Returns
    -------
    float or array : Δ'_bs [m⁻¹]
    """
    W = np.asarray(W, dtype=float)
    if r_s == 0.0 or L_q == 0.0 or L_p == 0.0:
        return np.zeros_like(W)
    W_thresh = float(rho_i)
    denom = W**2 + W_thresh**2
    return C_bs * (beta_pol / r_s) * (r_s / L_q) * (r_s / L_p) * np.where(denom > 0, W / denom, 0.0)


def delta_prime_eccd(j_eccd, r_s, B_theta_s, q_s, R0, W, sigma_eccd=None):
    """
    ECCD stabilization term in the MRE (Hegna–Callen form).

    Δ'_ECCD ≈ -μ₀ R₀ q_s / B_θ(r_s) * <j_ECCD> / W  * f_shape

    where f_shape accounts for the profile width relative to W.  For a
    Gaussian with σ_eccd >> W this saturates; for σ_eccd < W it scales ~1.

    Parameters
    ----------
    j_eccd    : float          peak ECCD current density [A/m²]
    r_s       : float          rational surface minor radius [m]
    B_theta_s : float          poloidal field at rational surface [T]
    q_s       : float          safety factor at rational surface (≈ 2)
    R0        : float          major radius [m]
    W         : float or array island half-width [m]
    sigma_eccd: float or None  1-σ width of ECCD deposition [m]; None → Gaussian ignored

    Returns
    -------
    float or array : Δ'_ECCD [m⁻¹]  (negative = stabilizing)
    """
    W = np.asarray(W)
    prefactor = -MU0 * R0 * q_s / B_theta_s
    if sigma_eccd is not None and np.isfinite(sigma_eccd) and sigma_eccd > 0:
        # Effective island-width averaged drive (Gaussian profile)
        f_shape = np.where(W > 0, np.tanh(W / sigma_eccd), 0.0)
        return prefactor * j_eccd * sigma_eccd * f_shape / W
    else:
        return prefactor * j_eccd / W


def seed_island_width_from_Brtilde(Br_tilde, r_wall, r_s, B_theta_s, m=2):
    """
    Seed island half-width from radial field fluctuation at the wall.

    Cylindrical extrapolation:
      Ψ̃(r_s) = |B̃_r(r_wall)| * r_wall / (m-1) * (r_s / r_wall)^m
    Island half-width (La Haye convention):
      W_seed = 4 √( r_s |Ψ̃(r_s)| / (m B_θ(r_s)) )

    Parameters
    ----------
    Br_tilde  : float or array  radial field fluctuation at wall [T]
    r_wall    : float           probe / wall minor radius [m]
    r_s       : float           q=m/n surface minor radius [m]
    B_theta_s : float           poloidal field at rational surface [T]
    m         : int             poloidal mode number

    Returns
    -------
    W_seed : float or array  seed island half-width [m]
    """
    Br_tilde = np.asarray(Br_tilde)
    psi_tilde_s = np.abs(Br_tilde) * r_wall / (m - 1) * (r_s / r_wall)**m
    W_seed = 4.0 * np.sqrt(r_s * psi_tilde_s / (m * B_theta_s))
    return W_seed


def B_theta_at_surface(Ip_A, r_s, kappa=1.8):
    """
    Approximate poloidal field at the rational surface (shaped-cylinder model).

    B_θ(r_s) ≈ μ₀ I_p / (2π r_s κ)

    Parameters
    ----------
    Ip_A  : float  plasma current [A]
    r_s   : float  rational surface minor radius [m]
    kappa : float  elongation (DIII-D typical ≈ 1.8)

    Returns
    -------
    float [T]
    """
    return MU0 * Ip_A / (2 * np.pi * r_s * kappa)


def find_q_surface(qpsi, aminor, q_target=2.0):
    """
    Find the minor radius r where q(r) = q_target via linear interpolation.

    Parameters
    ----------
    qpsi     : 1-D array  safety factor profile on uniform rho grid (0 → 1)
    aminor   : float      minor radius [m]
    q_target : float      target q value (default 2.0)

    Returns
    -------
    r_s : float or None  minor radius [m] of q=q_target surface
    """
    qpsi = np.asarray(qpsi, dtype=float)
    rho = np.linspace(0.0, 1.0, len(qpsi))
    crossings = np.where(np.diff(np.sign(qpsi - q_target)))[0]
    if len(crossings) == 0:
        return None
    i = crossings[0]
    frac = (q_target - qpsi[i]) / (qpsi[i + 1] - qpsi[i])
    rho_s = rho[i] + frac * (rho[i + 1] - rho[i])
    return rho_s * aminor


def gradient_scale_length(profile, rho_grid, aminor, rho_s):
    """
    Compute the profile scale length L = -f / (df/dr) at rho_s.

    Parameters
    ----------
    profile  : 1-D array  profile values on rho_grid
    rho_grid : 1-D array  normalized radial grid (0–1)
    aminor   : float      minor radius [m]
    rho_s    : float      normalized radius of rational surface

    Returns
    -------
    L : float  scale length [m]  (positive)
    """
    r = rho_grid * aminor
    dfddr = np.gradient(profile, r)
    f_at_s = np.interp(rho_s, rho_grid, profile)
    dfddr_at_s = np.interp(rho_s, rho_grid, dfddr)
    if dfddr_at_s == 0:
        return np.inf
    return -f_at_s / dfddr_at_s
