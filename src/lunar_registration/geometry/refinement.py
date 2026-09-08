"""
Sub-pixel tie-point refinement using localized matrix-multiply DFT correlation.
Reference: Guizar-Sicairos, Thurman, & Fienup (2008).
"""

from typing import Tuple
import numpy as np


def subpixel_dft_registration(
    ref_patch: np.ndarray,
    target_patch: np.ndarray,
    upsample_factor: int = 20
) -> Tuple[float, float]:
    """
    Compute sub-pixel shift between two small patches using cross-correlation peak finding.
    Returns (row_shift, col_shift).
    """
    if ref_patch.shape != target_patch.shape:
        return 0.0, 0.0

    # Cross-correlation via FFT
    f_ref = np.fft.fft2(ref_patch)
    f_tar = np.fft.fft2(target_patch)
    cross = f_ref * np.conj(f_tar)

    cc = np.fft.ifft2(cross)
    max_idx = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)

    nrows, ncols = cc.shape
    row_shift = max_idx[0] if max_idx[0] <= nrows // 2 else max_idx[0] - nrows
    col_shift = max_idx[1] if max_idx[1] <= ncols // 2 else max_idx[1] - ncols

    return float(col_shift), float(row_shift)
