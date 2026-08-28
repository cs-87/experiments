"""
The matched-filter bank, evaluated densely over a whole frame.

Two facts make a full-frame search cheap enough to be the default.

First, the DWT embedding is a pixel-domain embedding in disguise. Orthonormal Haar
gives LL[u,v] = B * (mean of the B x B pixel block), B = 2^level, and synthesis
spreads a unit LL delta as 1/B into each of the B^2 pixels. So `LL += alpha*W` is
identical to adding (alpha/B) * W with every element replicated over its B x B
block. The detector can therefore reconstruct the exact LL plane the embedder wrote
into by box-decimating the frame -- no wavelet library, no patch extraction, and
crucially no need to know where the patches are.

Second, decimation has B^2 possible grid alignments and the embedder's patches sit
at content-dependent offsets, so the detector forms all B^2 phase planes. Together
they cover every integer pixel offset in the frame, which is the resolution the
correlation needs: measured per-patch BER is 0.052 at zero offset, 0.076 at half a
pixel, and 0.228 at one pixel, so integer search is sufficient and sub-pixel search
is a refinement rather than a requirement.

Correlating a plane against all K PRNs at every position is then one FFT of the
plane plus K multiply-and-inverse-FFTs. About 6 GFLOP per 1080p frame at level 1 --
milliseconds on a GPU, a second or two on a CPU.
"""

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except ImportError:                                          # pragma: no cover
    _HAVE_TORCH = False


def ll_block_size(level):
    """Pixels per PRN chip: a level-`level` Haar LL sample covers a B x B block."""
    return 1 << int(level)


def ll_planes(luma, level=1):
    """
    All B^2 phase-shifted Haar LL planes of a luma frame, B = 2^level.

    Returns {(phase_y, phase_x): plane}. Plane element [i, j] is the LL coefficient
    of the block starting at frame pixel (phase_y + i*B, phase_x + j*B), on the same
    scale pywt.wavedec2(..., 'haar', level=level)[0] would produce: B times the block
    mean, because each of the `level` orthonormal Haar stages contributes a factor
    sqrt(2) per axis.
    """
    B = ll_block_size(level)
    y = np.ascontiguousarray(luma, dtype=np.float64)
    out = {}
    for py in range(B):
        for px in range(B):
            a = y[py:, px:]
            h = (a.shape[0] // B) * B
            w = (a.shape[1] // B) * B
            if h == 0 or w == 0:
                continue
            blocks = a[:h, :w].reshape(h // B, B, w // B, B)
            out[(py, px)] = blocks.mean(axis=(1, 3)) * B
    return out


def frame_to_plane(y, x, level):
    """Frame pixel (y, x) -> (phase, index within that phase's plane)."""
    B = ll_block_size(level)
    return (y % B, x % B), (y // B, x // B)


def plane_to_frame(phase, i, j, level):
    """Inverse of frame_to_plane."""
    B = ll_block_size(level)
    return phase[0] + i * B, phase[1] + j * B


class PRNCorrelator:
    """
    c_k(y, x) = <plane[y:y+s, x:x+s], P_k> / s^2 at every position, plus the
    codeword-independent energy T = sum_k c_k^2.

    The normalisation is by s^2 = the PRN length, so with no whitening and perfect
    alignment c_k comes out as exactly ALPHA/sqrt(32) * s_k -- the same units the
    signal model uses, which keeps every threshold interpretable.

    Energy is accumulated in chunks rather than materialising all K correlation maps:
    at 1080p level 1 the full stack is 265 MB per phase, the accumulated energy is
    2 MB, and only the peaks ever need their individual c_k.
    """

    def __init__(self, prns, device=None, chunk=8):
        prns = np.asarray(prns, dtype=np.float64)
        if prns.ndim != 3 or prns.shape[1] != prns.shape[2]:
            raise ValueError(f"prns must be (K, s, s), got {prns.shape}")
        self.prns = prns
        self.n_prns, self.side, _ = prns.shape
        self.n = self.side * self.side
        self.flat = prns.reshape(self.n_prns, self.n)
        self.chunk = int(chunk)

        self.torch = _HAVE_TORCH and (device != "cpu")
        if self.torch:
            self.device = torch.device(
                device if device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu"))
            self._t_prns = torch.as_tensor(prns, dtype=torch.float32, device=self.device)
        else:
            self.device = "cpu"
        self._cache = {}

    def __repr__(self):
        return (f"PRNCorrelator(K={self.n_prns}, side={self.side}, "
                f"device={self.device}, chunk={self.chunk})")

    # -- direct evaluation at known positions -------------------------------------

    def at(self, plane, positions):
        """
        (M, K) evidence for M (i, j) plane positions. Positions whose window would
        overhang are dropped, and the surviving positions are returned alongside so
        the caller never has to guess which row is which.
        """
        s, H, W = self.side, *plane.shape
        keep = [(i, j) for (i, j) in positions if 0 <= i <= H - s and 0 <= j <= W - s]
        if not keep:
            return np.zeros((0, self.n_prns)), []
        windows = np.stack([plane[i:i + s, j:j + s].ravel() for i, j in keep])
        return (windows @ self.flat.T) / self.n, keep

    # -- dense evaluation ----------------------------------------------------------

    def _templates_np(self, shape):
        key = ("np", shape)
        if key not in self._cache:
            pad = np.zeros((self.n_prns,) + shape)
            pad[:, :self.side, :self.side] = self.prns
            self._cache[key] = np.conj(np.fft.rfft2(pad, axes=(1, 2)))
        return self._cache[key]

    def _templates_torch(self, shape):
        key = ("torch", shape)
        if key not in self._cache:
            pad = torch.zeros((self.n_prns,) + shape, dtype=torch.float32,
                              device=self.device)
            pad[:, :self.side, :self.side] = self._t_prns
            self._cache[key] = torch.conj(torch.fft.rfft2(pad, dim=(1, 2)))
        return self._cache[key]

    def dense(self, plane, want_maps=False):
        """
        Cross-correlate `plane` against every PRN at every position.

        Returns (energy, maps). `energy` is sum_k c_k^2 with the same shape as
        `plane`; `maps` is the (K, H, W) stack only if want_maps, else None. Both are
        circular beyond position (H-s, W-s) -- callers must not read there, and
        valid_shape() says where the boundary is.
        """
        H, W = plane.shape
        if self.torch:
            x = torch.as_tensor(plane, dtype=torch.float32, device=self.device)
            X = torch.fft.rfft2(x)
            tmpl = self._templates_torch((H, W))
            energy = torch.zeros((H, W), dtype=torch.float32, device=self.device)
            maps = [] if want_maps else None
            for a in range(0, self.n_prns, self.chunk):
                b = min(a + self.chunk, self.n_prns)
                c = torch.fft.irfft2(X.unsqueeze(0) * tmpl[a:b], s=(H, W)) / self.n
                energy += (c ** 2).sum(0)
                if want_maps:
                    maps.append(c.cpu().numpy())
            return (energy.cpu().numpy(),
                    np.concatenate(maps, axis=0) if want_maps else None)

        X = np.fft.rfft2(plane)
        tmpl = self._templates_np((H, W))
        energy = np.zeros((H, W))
        maps = [] if want_maps else None
        for a in range(0, self.n_prns, self.chunk):
            b = min(a + self.chunk, self.n_prns)
            c = np.fft.irfft2(X[None] * tmpl[a:b], s=(H, W), axes=(1, 2)) / self.n
            energy += (c ** 2).sum(0)
            if want_maps:
                maps.append(c)
        return energy, (np.concatenate(maps, axis=0) if want_maps else None)

    def valid_shape(self, plane_shape):
        """Positions at or beyond this are circular wrap-around, not real overlap."""
        return plane_shape[0] - self.side + 1, plane_shape[1] - self.side + 1
