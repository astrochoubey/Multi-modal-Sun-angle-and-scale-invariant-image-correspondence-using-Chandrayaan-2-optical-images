"""
Sub-pixel tie-point refinement using localized matrix-multiply DFT correlation.
Reference: Guizar-Sicairos, Thurman, & Fienup (2008), Optics Letters 33(2), 156-158.
"""

from typing import Tuple
import numpy as np


def subpixel_dft_registration(
    ref_patch: np.ndarray,
    target_patch: np.ndarray,
    upsample_factor: int = 20
) -> Tuple[float, float]:
    """
    Compute sub-pixel shift between two small patches using matrix-multiply DFT upsampling.

    Parameters
    ----------
    ref_patch : np.ndarray
        Reference template patch (2D float or uint8).
    target_patch : np.ndarray
        Target warped patch (2D float or uint8).
    upsample_factor : int, optional
        Upsampling factor (default 20 gives 0.05 pixel precision).

    Returns
    -------
    Tuple[float, float]
        (col_shift, row_shift) representing (dx, dy) sub-pixel translation.
    """
    if ref_patch.shape != target_patch.shape:
        return 0.0, 0.0

    ref_f = ref_patch.astype(np.float32)
    tar_f = target_patch.astype(np.float32)

    # Zero-mean normalization to handle slight photometric shifts
    ref_f = ref_f - np.mean(ref_f)
    tar_f = tar_f - np.mean(tar_f)

    nrows, ncols = ref_f.shape
    if nrows < 4 or ncols < 4:
        return 0.0, 0.0

    # Cross-correlation via standard FFT
    f_ref = np.fft.fft2(ref_f)
    f_tar = np.fft.fft2(tar_f)
    cross = f_ref * np.conj(f_tar)

    cc = np.fft.ifft2(cross)
    max_idx = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)

    row_shift = float(max_idx[0] if max_idx[0] <= nrows // 2 else max_idx[0] - nrows)
    col_shift = float(max_idx[1] if max_idx[1] <= ncols // 2 else max_idx[1] - ncols)

    if upsample_factor > 1:
        # Matrix-multiply DFT in neighborhood around the integer peak
        row_shift_up = round(row_shift * upsample_factor) / upsample_factor
        col_shift_up = round(col_shift * upsample_factor) / upsample_factor
        dftshift = np.fix(np.ceil(upsample_factor * 1.5) / 2)

        roff = dftshift - row_shift_up * upsample_factor
        coff = dftshift - col_shift_up * upsample_factor

        # Kernel for columns
        c_freq = np.fft.ifftshift(np.arange(ncols) - np.floor(ncols / 2))
        c_space = np.arange(np.ceil(upsample_factor * 1.5)) - coff
        kernc = np.exp((-1j * 2 * np.pi / (ncols * upsample_factor)) * np.outer(c_freq, c_space))

        # Kernel for rows
        r_freq = np.fft.ifftshift(np.arange(nrows) - np.floor(nrows / 2))
        r_space = np.arange(np.ceil(upsample_factor * 1.5)) - roff
        kernr = np.exp((-1j * 2 * np.pi / (nrows * upsample_factor)) * np.outer(r_space, r_freq))

        cc_up = np.abs(kernr @ cross @ kernc)
        max_up = np.unravel_index(np.argmax(cc_up), cc_up.shape)

        row_shift = float(row_shift_up + (max_up[0] - dftshift) / upsample_factor)
        col_shift = float(col_shift_up + (max_up[1] - dftshift) / upsample_factor)

    return float(col_shift), float(row_shift)
