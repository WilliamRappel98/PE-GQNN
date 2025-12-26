import numpy as np
import torch
from scipy.stats import norm


def probit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the probit (inverse normal CDF) transformation of probabilities.

    Parameters
    ----------
    p : torch.Tensor
        Input probabilities. Must be within [0, 1].
    eps : float, optional
        Small value to avoid numerical issues with 0 or 1 inputs. Defaults to 1e-6.

    Returns
    -------
    torch.Tensor
        The input probabilities transformed to the probit scale.
    """
    p_clamped = torch.clamp(p, eps, 1 - eps)
    return torch.distributions.Normal(0.0, 1.0).icdf(p_clamped)


def mpe(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: torch.Tensor | None = None,
    var_pred: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the Mean Pinball Error (MPE). This can operate in two modes:
    1) If `tau` is provided, it computes the pinball loss directly.
    2) If `tau` is not provided, it samples `tau` from a uniform distribution
       and computes the pinball loss using quantiles from a normal distribution.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth values.
    y_pred : torch.Tensor
        Predicted values.
    tau : torch.Tensor | None, optional
        Quantile levels. If None, quantiles are sampled uniformly. Defaults to None.
    var_pred : torch.Tensor | None, optional
        Predicted variance for the normal distribution (if `tau` is None).
        Defaults to None.

    Returns
    -------
    torch.Tensor
        The mean pinball error.
    """
    if tau is not None:
        delta = y_true - y_pred
        pinball = torch.where(delta >= 0, tau * delta, (tau - 1) * delta)
        return pinball.mean()
    else:
        tau = torch.rand(y_true.shape)
        quantiles = norm.ppf(tau, loc=y_pred, scale=np.sqrt(var_pred))
        delta = y_true - quantiles
        pinball = torch.where(delta >= 0, tau * delta, (tau - 1) * delta)
        return pinball.mean()


def mav(y_true: np.ndarray | list[float]) -> float:
    """
    Calculate the Mean Absolute Value (MAV) of an array or list.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground truth values.

    Returns
    -------
    float
        The mean absolute value of the input.
    """
    y_true_arr = np.array(y_true)
    return float(np.mean(np.abs(y_true_arr)))


def coverage(
    y_true: np.ndarray | list[float],
    linf: np.ndarray | list[float],
    lsup: np.ndarray | list[float],
) -> float:
    """
    Compute the coverage probability of predictions. Given lower and upper
    prediction bounds, it calculates the fraction of true values that fall
    within those bounds.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground truth values.
    linf : np.ndarray | list[float]
        Lower prediction bounds.
    lsup : np.ndarray | list[float]
        Upper prediction bounds.

    Returns
    -------
    float
        The coverage probability, i.e., mean of (linf <= y_true <= lsup).
    """
    y_true_arr = np.array(y_true)
    linf_arr = np.array(linf)
    lsup_arr = np.array(lsup)
    return float(np.mean((y_true_arr >= linf_arr) & (y_true_arr <= lsup_arr)))


def smis(
    y_true: np.ndarray | list[float],
    linf: np.ndarray | list[float],
    lsup: np.ndarray | list[float],
    scale: float = -1,
    alpha: float = 0.05,
) -> float:
    """
    Compute the scaled Mean Interval Score (sMIS). If `scale` is -1, it uses
    the Mean Absolute Value (MAV) of `y_true` as the scale.

    Interval Score (IS):
    IS = (lsup - linf)
         + (2 / alpha) * (linf - y_true) * 1(y_true < linf)
         + (2 / alpha) * (y_true - lsup) * 1(y_true > lsup)

    The score is then averaged (MIS) and optionally scaled by `scale` (sMIS).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground truth values.
    linf : np.ndarray | list[float]
        Lower prediction bounds.
    lsup : np.ndarray | list[float]
        Upper prediction bounds.
    scale : float, optional
        Scaling value for the final MIS. If -1, use MAV of `y_true`.
        Defaults to -1.
    alpha : float, optional
        Significance level (controls penalty for out-of-bounds predictions).
        Defaults to 0.05.

    Returns
    -------
    float
        The scaled Mean Interval Score (sMIS).
    """
    y_true_arr = np.array(y_true)
    linf_arr = np.array(linf)
    lsup_arr = np.array(lsup)

    if scale == -1:
        scale = mav(y_true_arr)

    is_score = (
        (lsup_arr - linf_arr)
        + (2 / alpha) * (linf_arr - y_true_arr) * (y_true_arr < linf_arr).astype(int)
        + (2 / alpha) * (y_true_arr - lsup_arr) * (y_true_arr > lsup_arr).astype(int)
    )

    mis = np.mean(is_score)
    smis_value = mis / scale
    return float(smis_value)
