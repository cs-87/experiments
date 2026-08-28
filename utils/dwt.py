from dataclasses import dataclass
import pywt

WAVELET = "haar"
LEVEL = 1


class DWTLevel:
    def __init__(self, LH, HL, HH):
        self.LH = LH
        self.HL = HL
        self.HH = HH


class DWTCoeffs:
    def __init__(self, LL, levels):
        self.LL = LL
        self.levels = levels


def get_dwt_coeff(array, level=LEVEL):

    coeffs = pywt.wavedec2(array, WAVELET, level=level)
    # Convert the coefficients to the custom class structure
    dwt_coeffs = DWTCoeffs(LL=coeffs[0], levels=[])
    for level_coeffs in coeffs[1:]:
        dwt_coeffs.levels.append(DWTLevel(*level_coeffs))
    return dwt_coeffs


def reconstruct_frame(coeffs: DWTCoeffs):
    LL = coeffs.LL
    levels = [(level.LH, level.HL, level.HH) for level in coeffs.levels]
    coeffs = [LL] + levels
    reconstructed_frame = pywt.waverec2(coeffs, WAVELET)
    return reconstructed_frame
