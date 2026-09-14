"""
Utility functions for PIA-IVIM.

Includes data loading, rRMSE evaluation metrics (AAPM competition),
bi-exponential IVIM model, NLLS fitting, and synthetic batch generation.
"""
import numpy as np
from scipy.optimize import curve_fit
import torch
from tqdm import tqdm


def read_data(file_dir, fname, i):
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)
    return data


def rRMSE(x, y, t, is_f=False):
    """Per-parameter rRMSE following AAPM Grand Challenge Eq. 4.

    For f (is_f=True): C* = 1 (absolute RMSE, since f is a ratio)
        R = sqrt( (1/N) * sum( (x_j - y_j)^2 ) )

    For Dt, D* (is_f=False): C* = p* (per-voxel relative RMSE)
        R = sqrt( (1/N) * sum( ((x_j - y_j) / y_j)^2 ) )

    Returns (tumor_rRMSE, nontumor_rRMSE), excluding air (tissue == 1).
    """
    Nx, Ny = x.shape

    t_tmp = np.reshape(t, (Nx * Ny,))
    tumor_indice = np.argwhere(t_tmp == 8).ravel()
    non_tumor_indice = np.argwhere(t_tmp != 8).ravel()
    non_air_indice = np.argwhere(t_tmp != 1).ravel()
    non_tumor_air_indice = np.intersect1d(non_tumor_indice, non_air_indice)

    x_tmp = np.reshape(x, (Nx * Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]

    y_tmp = np.reshape(y, (Nx * Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]

    if is_f:
        # C* = 1: absolute RMSE
        z_t = np.sqrt(np.mean(np.square(x_t - y_t))) if len(x_t) > 0 else 0.0
        z_nt = np.sqrt(np.mean(np.square(x_nt - y_nt))) if len(x_nt) > 0 else 0.0
    else:
        # C* = p* (ground truth): per-voxel relative RMSE
        y_t_safe = np.clip(np.abs(y_t), 1e-10, None)
        y_nt_safe = np.clip(np.abs(y_nt), 1e-10, None)
        z_t = np.sqrt(np.mean(np.square((x_t - y_t) / y_t_safe))) if len(x_t) > 0 else 0.0
        z_nt = np.sqrt(np.mean(np.square((x_nt - y_nt) / y_nt_safe))) if len(x_nt) > 0 else 0.0

    return z_t, z_nt


def rRMSE_per_case(x_f, x_dt, x_ds, y_f, y_dt, y_ds, t):
    R_f_t, R_f_nt = rRMSE(x_f, y_f, t, is_f=True)
    R_Dt_t, R_Dt_nt = rRMSE(x_dt, y_dt, t)
    R_Ds_t, R_Ds_nt = rRMSE(x_ds, y_ds, t)

    z = (R_f_t + R_Dt_t + R_Ds_t) / 3 + (R_f_nt + R_Dt_nt) / 2
    z_t = (R_f_t + R_Dt_t + R_Ds_t) / 3

    return z, z_t


def rRMSE_all_cases(x_f, x_dt, x_ds, y_f, y_dt, y_ds, t):
    z = np.empty([x_f.shape[2]])
    z_t = np.empty([x_f.shape[2]])

    for i in range(x_f.shape[2]):
        z[i], z_t[i] = rRMSE_per_case(
            x_f[:, :, i], x_dt[:, :, i], x_ds[:, :, i],
            y_f[:, :, i], y_dt[:, :, i], y_ds[:, :, i],
            t[:, :, i]
        )

    return np.average(z), np.average(z_t)


def rRMSE_per_param(x_f, x_dt, x_ds, y_f, y_dt, y_ds, t):
    """Compute per-parameter rRMSE for a single case.
    
    Returns individual rRMSE values for f, Dt, D* in both tumor and non-tumor regions,
    plus the composite scores.
    
    Returns:
        dict with keys: 'f_tumor', 'f_nontumor', 'Dt_tumor', 'Dt_nontumor',
                        'Dstar_tumor', 'Dstar_nontumor', 'composite_total', 'composite_tumor'
    """
    R_f_t, R_f_nt = rRMSE(x_f, y_f, t, is_f=True)
    R_Dt_t, R_Dt_nt = rRMSE(x_dt, y_dt, t)
    R_Ds_t, R_Ds_nt = rRMSE(x_ds, y_ds, t)
    
    composite_total = (R_f_t + R_Dt_t + R_Ds_t) / 3 + (R_f_nt + R_Dt_nt) / 2
    composite_tumor = (R_f_t + R_Dt_t + R_Ds_t) / 3
    
    return {
        'f_tumor': float(R_f_t), 'f_nontumor': float(R_f_nt),
        'Dt_tumor': float(R_Dt_t), 'Dt_nontumor': float(R_Dt_nt),
        'Dstar_tumor': float(R_Ds_t), 'Dstar_nontumor': float(R_Ds_nt),
        'composite_total': float(composite_total),
        'composite_tumor': float(composite_tumor),
    }


def funcBiExp(b, f, Dt, Ds):
    """Standard bi-exponential IVIM model.

    Units (canonical):
        b:  s/mm^2  (e.g. 0, 5, 50, ... 1000)
        Dt: mm^2/s  (e.g. ~0.001)
        Ds: mm^2/s  (e.g. ~0.01)
        f:  dimensionless fraction [0, 1]
    """
    return (1. - f) * np.exp(-1. * Dt * b) + f * np.exp(-1. * Ds * b)


def fit_biExponential_model(arr3D_img, arr1D_b):
    """Voxel-wise NLLS bi-exponential IVIM fitting with segmented initialization.

    Uses the standard two-step segmented approach for robust initialization:
    1) Estimate Dt from high b-values (b >= 200) via log-linear regression
    2) Derive f and D* initial guesses from the residual
    3) Full bi-exponential fit with Trust Region Reflective (matching AAPM GC baseline)

    On convergence failure, falls back to the segmented estimates rather than
    zeros (which would artificially inflate rRMSE).
    """
    arr2D_coordBody = np.argwhere(arr3D_img[:, :, 0] > 0)
    arr2D_fFitted = np.zeros_like(arr3D_img[:, :, 0])
    arr2D_DtFitted = np.zeros_like(arr3D_img[:, :, 0])
    arr2D_DsFitted = np.zeros_like(arr3D_img[:, :, 0])

    b_fit = arr1D_b[1:] - arr1D_b[0]  # e.g. [5, 50, 100, 200, 500, 800, 1000]
    high_b_mask = b_fit >= 200  # indices for high b-values

    for arr1D_coord in arr2D_coordBody:
        ix, iy = arr1D_coord[0], arr1D_coord[1]
        s0 = arr3D_img[ix, iy, 0]
        if s0 <= 0:
            continue
        signal_norm = arr3D_img[ix, iy, 1:] / s0

        # --- Segmented initialization ---
        # Step 1: linear fit on high b-values to estimate Dt
        # At high b, perfusion signal decays away: S/S0 ~ (1-f)*exp(-b*Dt)
        # => ln(S/S0) ~ ln(1-f) - b*Dt
        try:
            high_signal = np.clip(signal_norm[high_b_mask], 1e-10, None)
            log_signal = np.log(high_signal)
            high_b = b_fit[high_b_mask]
            coeffs = np.polyfit(high_b, log_signal, 1)
            Dt_init = np.clip(-coeffs[0], 1e-5, 2.9e-3)
            f_init = np.clip(1.0 - np.exp(coeffs[1]), 0.01, 0.99)
            Ds_init = np.clip(Dt_init * 10, 3.0e-3, 0.1)
        except Exception:
            # Fallback to standard initial guess
            f_init, Dt_init, Ds_init = 0.15, 1.5e-3, 8e-3

        # --- Full bi-exponential fit ---
        try:
            popt, _ = curve_fit(
                funcBiExp,
                b_fit,
                signal_norm,
                p0=(f_init, Dt_init, Ds_init),
                bounds=([0, 0, 3.0e-3], [1, 2.9e-3, 0.1]),
                method='trf'
            )
        except Exception:
            # Use segmented estimates as fallback (not zeros)
            popt = [f_init, Dt_init, Ds_init]

        arr2D_fFitted[ix, iy] = popt[0]
        arr2D_DtFitted[ix, iy] = popt[1]
        arr2D_DsFitted[ix, iy] = popt[2]

    return np.concatenate((
        arr2D_fFitted[:, :, np.newaxis],
        arr2D_DtFitted[:, :, np.newaxis],
        arr2D_DsFitted[:, :, np.newaxis]
    ), axis=2)


def get_batch(batch_size=16, noise_sdt=0.01):
    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

    Dt = np.random.uniform(0.0001, 0.0025, batch_size)
    D_star = np.random.uniform(0.002, 0.065, batch_size)
    f = np.random.uniform(0.005, 0.4, batch_size)

    signal = np.zeros((batch_size, len(b_values)), dtype=float)
    for sample in range(batch_size):
        for ctr, b in enumerate(b_values):
            signal[sample, ctr] = (1 - f[sample]) * np.exp(-b * Dt[sample]) + f[sample] * np.exp(-b * D_star[sample])

    noise_re = np.random.normal(0, noise_sdt, signal.shape)
    noise_im = np.random.normal(0, noise_sdt, signal.shape)
    noisy = np.sqrt((signal + noise_re)**2 + noise_im**2)

    return (torch.from_numpy(noisy).float(),
            torch.from_numpy(f.T).float(),
            torch.from_numpy(Dt.T).float(),
            torch.from_numpy(D_star.T).float(),
            torch.from_numpy(signal).float())