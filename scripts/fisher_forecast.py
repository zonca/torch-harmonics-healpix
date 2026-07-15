"""Fisher forecast for Test 4: theoretical lower bounds on σ(r) and σ(τ).

Fisher Matrix Formalism
=======================
The Fisher information matrix provides a rigorous lower bound on the
covariance of any unbiased estimator via the Cramér-Rao inequality:

    Cov(θ_i, θ_j) ≥ (F⁻¹)_ij

where the Fisher matrix F is defined as:

    F_ij = Σ_ℓ ∑_X  [∂C_ℓ^X/∂θ_i] [∂C_ℓ^X/∂θ_j] / σ²(C_ℓ^X)

with X ∈ {EE, BB}, θ = [r, τ], and the variance of each observed
power spectrum bandpower is:

    σ²(Ĉ_ℓ^X) = (2 / (2ℓ + 1)) * (C_ℓ^X + N_ℓ)² / f_sky

This is the cosmic variance + noise limit: even an optimal estimator
cannot beat these uncertainties. The Fisher forecast thus provides
the best-case (theoretical lower bound) performance for joint r/τ
estimation, against which the neural network's empirical errors can
be compared.

Key physical intuition:
- r is constrained primarily by BB (tensor B-modes), with some EE contribution.
- τ is constrained primarily by EE (reionization bump at low ℓ).
- The correlation between r and τ errors arises because both affect
  large-scale polarization.
"""

import argparse

import numpy as np

from torch_harmonics_healpix.data_generation_test4 import (
    generate_camb_spectra_r_tau,
)


# Fixed high lmax for CAMB calculations. CAMB's set_for_lmax() changes
# internal accuracy parameters (max_eta_k, etc.), producing *different*
# C_ℓ values at the same ℓ depending on the lmax setting. To make Fisher
# bounds comparable across NSIDEs, all spectra must be computed with the
# same CAMB configuration. We use a fixed high lmax and truncate.
LMAX_CALC_FIXED = 500


def _generate_spectra_fixed_lmax(
    r: float, tau: float, lmax_out: int, lmax_calc: int = LMAX_CALC_FIXED
) -> tuple[np.ndarray, np.ndarray]:
    """Generate CAMB spectra at a fixed high lmax, then truncate to lmax_out.

    This ensures consistent C_ℓ values across different output lmax values,
    which is essential for comparing Fisher bounds across NSIDEs.
    """
    cl_ee, cl_bb = generate_camb_spectra_r_tau(r, tau, lmax_calc)
    return cl_ee[: lmax_out + 1], cl_bb[: lmax_out + 1]


def compute_fisher_matrix(
    r: float,
    tau: float,
    lmax: int = 47,
    nside: int = 16,
    noise_std_uK: float = 6.0,
    f_sky: float = 1.0,
    lmax_calc: int = LMAX_CALC_FIXED,
) -> np.ndarray:
    """Compute the 2×2 Fisher information matrix for joint (r, τ) estimation.

    Uses both EE and BB power spectra. Derivatives of C_ℓ with respect to
    r and τ are computed via central finite differences. The variance of
    each bandpower includes cosmic variance and white noise, with a sky
    fraction correction.

    Args:
        r: Fiducial tensor-to-scalar ratio.
        tau: Fiducial optical depth to reionization.
        lmax: Maximum multipole moment.
        nside: HEALPix NSIDE parameter (determines pixel count for noise).
        noise_std_uK: White noise standard deviation in μK (per pixel).
            If converting from μK-arcmin, use: noise_uK = noise_arcmin / sqrt(pixel_area_arcmin²).
        f_sky: Observed sky fraction.
        lmax_calc: Fixed lmax for CAMB calculation (ensures consistent
            C_ℓ across NSIDEs). Default 500.

    Returns:
        2×2 Fisher information matrix, F, with indices corresponding to
        θ = [r, τ]. F[0,0] = Fisher info on r, F[1,1] = on τ,
        F[0,1] = F[1,0] = cross term.
    """
    # Step sizes for central differences
    dr = 1e-5
    dtau = 1e-4

    # Fiducial spectra — computed at fixed high lmax, truncated to lmax
    cl_ee_fid, cl_bb_fid = _generate_spectra_fixed_lmax(
        r, tau, lmax, lmax_calc
    )

    # Spectra at r ± dr (tau fixed)
    cl_ee_rplus, cl_bb_rplus = _generate_spectra_fixed_lmax(
        r + dr, tau, lmax, lmax_calc
    )
    cl_ee_rminus, cl_bb_rminus = _generate_spectra_fixed_lmax(
        max(r - dr, 0), tau, lmax, lmax_calc
    )

    # Spectra at tau ± dtau (r fixed)
    cl_ee_tauplus, cl_bb_tauplus = _generate_spectra_fixed_lmax(
        r, tau + dtau, lmax, lmax_calc
    )
    cl_ee_tauminus, cl_bb_tauminus = _generate_spectra_fixed_lmax(
        r, tau - dtau, lmax, lmax_calc
    )

    # Central difference derivatives
    dcl_ee_dr = (cl_ee_rplus - cl_ee_rminus) / (2 * dr)
    dcl_bb_dr = (cl_bb_rplus - cl_bb_rminus) / (2 * dr)
    dcl_ee_dtau = (cl_ee_tauplus - cl_ee_tauminus) / (2 * dtau)
    dcl_bb_dtau = (cl_bb_tauplus - cl_bb_tauminus) / (2 * dtau)

    # Noise power: white noise N_ℓ = σ² * 4π / N_pix (flat in ℓ)
    npix = 12 * nside**2
    n_ell = noise_std_uK**2 * 4.0 * np.pi / npix

    # Sum over ℓ = 2 to lmax
    fisher = np.zeros((2, 2))

    for ell in range(2, lmax + 1):
        # Variance of each bandpower (cosmic variance + noise) / f_sky
        # σ²(Ĉ_ℓ^EE) = (2/(2ℓ+1)) * (C_ℓ^EE + N_ℓ)² / f_sky
        # σ²(Ĉ_ℓ^BB) = (2/(2ℓ+1)) * (C_ℓ^BB + N_ℓ)² / f_sky
        var_ee = (2.0 / (2 * ell + 1)) * (cl_ee_fid[ell] + n_ell) ** 2 / f_sky
        var_bb = (2.0 / (2 * ell + 1)) * (cl_bb_fid[ell] + n_ell) ** 2 / f_sky

        # Derivative vectors at this ℓ: [d/d_r, d/d_tau]
        dc_dr = np.array([dcl_ee_dr[ell], dcl_bb_dr[ell]])
        dc_dtau = np.array([dcl_ee_dtau[ell], dcl_bb_dtau[ell]])
        var = np.array([var_ee, var_bb])

        # Fisher contribution: F_ij += Σ_X (dC^X/dθ_i)(dC^X/dθ_j) / σ²(C^X)
        # Sum over X = {EE, BB}
        for i, dc_di in enumerate([dc_dr, dc_dtau]):
            for j, dc_dj in enumerate([dc_dr, dc_dtau]):
                fisher[i, j] += np.sum(dc_di * dc_dj / var)

    return fisher


def fisher_forecast(
    r: float = 0.003,
    tau: float = 0.054,
    lmax: int = 47,
    nside: int = 16,
    noise_std_uK: float = 6.0,
    f_sky: float = 1.0,
) -> dict:
    """Compute Fisher forecast for joint (r, τ) estimation.

    Computes the Fisher information matrix, inverts it to obtain the
    Cramér-Rao lower bound on parameter covariances, and extracts
    1σ uncertainties and the correlation coefficient.

    Args:
        r: Fiducial tensor-to-scalar ratio.
        tau: Fiducial optical depth to reionization.
        lmax: Maximum multipole moment.
        nside: HEALPix NSIDE parameter.
        noise_std_uK: White noise standard deviation in μK (per pixel).
        f_sky: Observed sky fraction.

    Returns:
        Dictionary with keys:
            - 'sigma_r': 1σ uncertainty on r (Cramér-Rao lower bound).
            - 'sigma_tau': 1σ uncertainty on τ (Cramér-Rao lower bound).
            - 'correlation_r_tau': Correlation coefficient between r and τ errors.
            - 'fisher_matrix': The 2×2 Fisher information matrix.
            - 'covariance_matrix': The inverse Fisher matrix (parameter covariance).
    """
    fisher = compute_fisher_matrix(r, tau, lmax, nside, noise_std_uK, f_sky)
    covariance = np.linalg.inv(fisher)

    sigma_r = np.sqrt(covariance[0, 0])
    sigma_tau = np.sqrt(covariance[1, 1])
    correlation_r_tau = covariance[0, 1] / (sigma_r * sigma_tau)

    return {
        "sigma_r": sigma_r,
        "sigma_tau": sigma_tau,
        "correlation_r_tau": correlation_r_tau,
        "fisher_matrix": fisher,
        "covariance_matrix": covariance,
    }


def main():
    """Command-line interface for Fisher forecast of Test 4.

    Parses fiducial cosmology and experimental parameters, runs the
    Fisher forecast, and prints the Cramér-Rao lower bounds on σ(r)
    and σ(τ) along with their correlation and fractional errors.

    Noise conversion: μK-arcmin → μK per pixel. For a HEALPix pixel
    with area Ω_pix = 4π / N_pix steradians, the arcmin² area is
    Ω_pix * (180/π)² * 3600, and the per-pixel noise is:

        σ_pix [μK] = σ_arcmin [μK-arcmin] / sqrt(Ω_pix_arcmin²)
    """
    parser = argparse.ArgumentParser(
        description="Fisher forecast for Test 4: joint r/τ estimation lower bounds"
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.003,
        help="Fiducial tensor-to-scalar ratio (default: 0.003)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.054,
        help="Fiducial optical depth (default: 0.054)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=6.0,
        help="White noise level in μK-arcmin (default: 6)",
    )
    parser.add_argument(
        "--f_sky",
        type=float,
        default=1.0,
        help="Observed sky fraction (default: 1.0)",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Maximum multipole (default: 3*NSIDE-1)",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=16,
        help="HEALPix NSIDE (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for results",
    )

    args = parser.parse_args()

    if args.lmax is None:
        args.lmax = 3 * args.nside - 1

    # Convert noise from μK-arcmin to μK per HEALPix pixel
    # Pixel area in arcmin²:
    #   Ω_pix [sr] = 4π / N_pix
    #   Ω_pix [arcmin²] = Ω_pix [sr] * (180*60/π)²
    #   σ_pix = σ_arcmin / sqrt(Ω_pix_arcmin²)
    npix = 12 * args.nside**2
    pixel_area_sr = 4.0 * np.pi / npix
    pixel_area_arcmin2 = pixel_area_sr * (180.0 * 60.0 / np.pi) ** 2
    noise_std_uK = args.noise_std / np.sqrt(pixel_area_arcmin2)

    print("=" * 60)
    print("Fisher Forecast for Test 4: Joint r/τ Estimation")
    print("=" * 60)
    print(f"Fiducial r   = {args.r}")
    print(f"Fiducial τ   = {args.tau}")
    print(f"NSIDE        = {args.nside}")
    print(f"LMAX         = {args.lmax}")
    print(f"Noise        = {args.noise_std} μK-arcmin = {noise_std_uK:.4f} μK/pix")
    print(f"f_sky        = {args.f_sky}")
    print("-" * 60)

    result = fisher_forecast(
        r=args.r,
        tau=args.tau,
        lmax=args.lmax,
        nside=args.nside,
        noise_std_uK=noise_std_uK,
        f_sky=args.f_sky,
    )

    sigma_r = result["sigma_r"]
    sigma_tau = result["sigma_tau"]
    corr = result["correlation_r_tau"]

    print(f"\nFisher matrix:")
    print(result["fisher_matrix"])
    print(f"\nCovariance matrix (Cramér-Rao lower bound):")
    print(result["covariance_matrix"])
    print(f"\nResults:")
    print(f"  σ(r)       = {sigma_r:.6f}")
    print(f"  σ(τ)       = {sigma_tau:.6f}")
    print(f"  ρ(r, τ)    = {corr:.4f}")
    print(f"  r error    = {sigma_r / args.r * 100:.1f}%")
    print(f"  τ error    = {sigma_tau / args.tau * 100:.1f}%")
    print("=" * 60)

    if args.output:
        import json, os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        r_fisher_pct = float(sigma_r / args.r * 100)
        tau_fisher_pct = float(sigma_tau / args.tau * 100)
        config_label = f"fsky{args.f_sky}_noise{int(args.noise_std)}"
        result_data = {
            "test": "test4_fisher",
            "nside": args.nside,
            "config": config_label,
            "f_sky": args.f_sky,
            "noise_arcmin": args.noise_std,
            "noise_uK_per_pixel": float(noise_std_uK),
            "r_fiducial": args.r,
            "tau_fiducial": args.tau,
            "sigma_r": float(sigma_r),
            "sigma_tau": float(sigma_tau),
            "sigma_r_fisher": float(sigma_r),
            "sigma_tau_fisher": float(sigma_tau),
            "r_pct_error": r_fisher_pct,
            "tau_pct_error": tau_fisher_pct,
            "r_fisher_pct": r_fisher_pct,
            "tau_fisher_pct": tau_fisher_pct,
            "correlation_r_tau": float(corr),
            "lmax": args.lmax,
        }
        with open(args.output, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
