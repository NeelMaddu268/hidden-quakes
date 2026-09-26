"""Location PDF and formal errors from the grid-search misfit on the fine lattice.

Conventions (also returned by ``conventions()`` for the run record):

- **PDF.** The misfit at a node is ``sum_i w_i |r_i|`` with ``w_i = prob_i / sigma_i`` and the
  origin time at its best (weighted-median) value. Node mass is ``exp(-s * misfit)`` with
  ``s = pdfMisfitScale``. With ``s = 1`` (the ticket's ``exp(-misfit)``) this is the Laplace
  likelihood ``prod_i exp(-|r_i| / b_i)`` with scale ``b_i = sigma_i / prob_i`` (a Laplace variable
  with scale b has standard deviation sqrt(2) b), profiled over the origin time, with a uniform
  prior over the evaluated nodes; ``s`` rescales every ``b_i`` by ``1 / s``. The constant
  ``prod_i s / (2 b_i)`` cancels in the normalisation: node masses are
  ``exp(-s * (misfit - min misfit))`` divided by their sum, so they add up to 1. ``s`` changes the
  formal errors only; the hypocentre is the least-misfit node whatever ``s`` is.
- **Calibration.** For Gaussian pick noise with standard deviation sigma and ``s = 1``, the PDF's
  standard deviation is sqrt(2/pi)^(1/2) = 0.893 times the L1 estimate's scatter, so the "68%"
  errors cover about 63% (the synthetic test measures it). ``s = sqrt(2/pi)`` would calibrate them.
- **Moments.** The PDF mean and 3x3 covariance of ``(e, n, elevM)`` are taken over the node masses.
- **vErrM** is the standard deviation of the vertical marginal (a 68% half-width in 1D).
- **hErrM** is the semi-major axis of the horizontal marginal's confidence ellipse at
  ``errConfidence`` (0.68, fixed by docs/02): ``sqrt(chi2_2(errConfidence) * lambda_max)``, with
  ``chi2_2`` the chi-square quantile with 2 degrees of freedom (from scipy) and ``lambda_max`` the
  largest eigenvalue of the 2x2 horizontal covariance.
- **Grid floor.** A PDF whose mass sits on one node would give 0 m. The horizontal eigenvalue and
  the vertical variance are floored at the variance of a uniform distribution over one cell,
  ``spacing**2 / 12``, and the floor is flagged.
- **Face masses.** The PDF mass on each of the six faces of the evaluated fine region (one node
  layer each).
- **depthOnEdge** is set when more than ``depthOnEdgeMassFraction`` of the mass lies on the fine
  grid's top face or on its bottom face (each face checked on its own; docs/02). MAP positions
  on a volume boundary are recorded separately, including broad PDFs with small face mass.
- **Truncation.** The locator grows the evaluated region until no face that is not the volume's top
  or bottom face holds a node within ``pdfCutoff`` of the minimum. If a lateral volume face or the
  node budget (``maxPdfNodes``) stops it first, the PDF is truncated there: the locator reports
  hErrM and vErrM as None (docs/02 allows None) and ``pdfTruncated`` in its search record.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]
# Face names of a (z, n, e) box, in the order summarize_pdf reports them.
FACES = ("top", "bottom", "north", "south", "east", "west")


def chi2_2(confidence: float) -> float:
    """Chi-square quantile with 2 degrees of freedom at ``confidence`` (scipy, not tabulated)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    return float(chi2.ppf(confidence, df=2))


@dataclass(frozen=True, eq=False)
class PdfSummary:
    """Moments, formal errors and face masses of a location PDF on the fine lattice."""

    mean_e_m: float
    mean_n_m: float
    mean_elev_m: float
    cov: FloatArray  # 3x3 over (e, n, elevM), m^2, read-only
    h_err_m: float
    v_err_m: float
    h_err_floored: bool
    v_err_floored: bool
    face_mass: dict[str, float]  # FACES -> PDF mass on that face of the evaluated box
    map_on_volume_top: bool
    map_on_volume_bottom: bool
    depth_on_edge: bool
    max_node_mass: float
    n_nodes: int

    @property
    def top_face_mass(self) -> float:
        return self.face_mass["top"]

    @property
    def bottom_face_mass(self) -> float:
        return self.face_mass["bottom"]

    def to_record(self) -> dict[str, Any]:
        return {
            "meanEM": self.mean_e_m,
            "meanNM": self.mean_n_m,
            "meanElevM": self.mean_elev_m,
            "cov": self.cov.tolist(),
            "hErrM": self.h_err_m,
            "vErrM": self.v_err_m,
            "hErrFloored": self.h_err_floored,
            "vErrFloored": self.v_err_floored,
            "faceMass": dict(self.face_mass),
            "mapOnVolumeTop": self.map_on_volume_top,
            "mapOnVolumeBottom": self.map_on_volume_bottom,
            "depthOnEdge": self.depth_on_edge,
            "maxNodeMass": self.max_node_mass,
            "nNodes": self.n_nodes,
        }


def summarize_pdf(
    misfit: FloatArray,
    e_axis: FloatArray,
    n_axis: FloatArray,
    z_axis: FloatArray,
    *,
    spacing_h_m: float,
    spacing_z_m: float,
    misfit_scale: float,
    top_is_volume_top: bool,
    bottom_is_volume_bottom: bool,
    confidence: float,
    edge_fraction: float,
) -> PdfSummary:
    """PDF summary of a misfit box of shape ``(len(z_axis), len(n_axis), len(e_axis))``.

    The box is the evaluated fine region; its faces are the fine grid's faces. ``top_is_volume_top``
    / ``bottom_is_volume_bottom`` say whether its top / bottom face is the search volume's.
    """
    misfit = np.asarray(misfit, dtype=np.float64)
    shape = (z_axis.size, n_axis.size, e_axis.size)
    if misfit.shape != shape:
        raise ValueError(f"misfit shape {misfit.shape} does not match the axes {shape}")
    if not np.all(np.isfinite(misfit)):
        raise ValueError("misfit has non-finite values")
    if not misfit_scale > 0:
        raise ValueError(f"misfit_scale must be positive, got {misfit_scale}")
    mass = np.exp(-misfit_scale * (misfit - misfit.min()))
    mass /= mass.sum()

    # Moments about the MAP node, to keep the products small.
    iz, i_n, ie = np.unravel_index(int(np.argmin(misfit)), shape)
    de = e_axis - e_axis[ie]
    dn = n_axis - n_axis[i_n]
    dz = z_axis - z_axis[iz]
    m_z = mass.sum(axis=(1, 2))
    m_n = mass.sum(axis=(0, 2))
    m_e = mass.sum(axis=(0, 1))
    mu = np.array([m_e @ de, m_n @ dn, m_z @ dz])
    m_ne = mass.sum(axis=0)  # (n, e)
    m_ze = mass.sum(axis=1)  # (z, e)
    m_zn = mass.sum(axis=2)  # (z, n)
    second = np.empty((3, 3))
    second[0, 0] = m_e @ de**2
    second[1, 1] = m_n @ dn**2
    second[2, 2] = m_z @ dz**2
    second[0, 1] = second[1, 0] = dn @ m_ne @ de
    second[0, 2] = second[2, 0] = dz @ m_ze @ de
    second[1, 2] = second[2, 1] = dz @ m_zn @ dn
    cov = second - np.outer(mu, mu)
    cov = 0.5 * (cov + cov.T)
    cov.setflags(write=False)

    lam_max = float(np.linalg.eigvalsh(cov[:2, :2])[-1])
    h_floor = spacing_h_m**2 / 12.0
    v_floor = spacing_z_m**2 / 12.0
    h_floored = lam_max < h_floor
    v_var = float(cov[2, 2])
    v_floored = v_var < v_floor
    h_err = math.sqrt(chi2_2(confidence) * max(lam_max, h_floor))
    v_err = math.sqrt(max(v_var, v_floor))

    face_mass = {
        "top": float(m_z[-1]),
        "bottom": float(m_z[0]),
        "north": float(m_n[-1]),
        "south": float(m_n[0]),
        "east": float(m_e[-1]),
        "west": float(m_e[0]),
    }
    on_top = bool(top_is_volume_top and iz == shape[0] - 1)
    on_bottom = bool(bottom_is_volume_bottom and iz == 0)
    depth_on_edge = (
        face_mass["top"] > edge_fraction
        or face_mass["bottom"] > edge_fraction
    )
    return PdfSummary(
        mean_e_m=float(e_axis[ie] + mu[0]),
        mean_n_m=float(n_axis[i_n] + mu[1]),
        mean_elev_m=float(z_axis[iz] + mu[2]),
        cov=cov,
        h_err_m=h_err,
        v_err_m=v_err,
        h_err_floored=h_floored,
        v_err_floored=v_floored,
        face_mass=face_mass,
        map_on_volume_top=on_top,
        map_on_volume_bottom=on_bottom,
        depth_on_edge=depth_on_edge,
        max_node_mass=float(mass.max()),
        n_nodes=int(mass.size),
    )


def conventions(confidence: float, edge_fraction: float, misfit_scale: float) -> dict[str, Any]:
    """The uncertainty conventions as recorded in ``ProcessingRun.locator``."""
    return {
        "pdf": "node mass proportional to exp(-pdfMisfitScale * misfit); pdfMisfitScale 1 is the "
        "Laplace likelihood with scale sigma/prob per pick, profiled over the origin time "
        "(weighted median), uniform prior over the fine nodes; normalised to sum to 1 over the "
        "evaluated fine nodes. Formal errors only: the hypocentre is the least-misfit node",
        "pdfMisfitScale": misfit_scale,
        "calibration": "with Gaussian pick noise and pdfMisfitScale 1 the PDF sd is 0.893 x the "
        "estimate's scatter (the 68% errors cover ~63%); sqrt(2/pi) = 0.798 calibrates it; the "
        "synthetic test records the measured coverage (fracHWithinHErrM, fracVWithinVErrM)",
        "vErrM": "standard deviation of the vertical marginal (68% in 1D)",
        "hErrM": "sqrt(chi2_2(errConfidence) * lambda_max): semi-major axis of the horizontal "
        "marginal's confidence ellipse, lambda_max the largest eigenvalue of the 2x2 horizontal "
        "covariance",
        "errConfidence": confidence,
        "chi2_2": chi2_2(confidence),
        "gridFloor": "lambda_max and the vertical variance are floored at spacing^2 / 12 (flagged)",
        "depthOnEdge": f"> {edge_fraction} of the mass on the evaluated fine grid's top face or on "
        "its bottom face; MAP boundary positions are separate diagnostics",
        "depthOnEdgeMassFraction": edge_fraction,
        "truncation": "the evaluated region grows until no face other than the volume's top or "
        "bottom holds a node within pdfCutoff of the minimum; if a lateral volume face or "
        "maxPdfNodes stops it, hErrM and vErrM are None and the search record says pdfTruncated",
    }
