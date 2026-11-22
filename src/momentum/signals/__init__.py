"""Signals: compute the 12-1 momentum score and rank stocks cross-sectionally."""

from momentum.signals.momentum import MomentumSpec, Signal, compute_signal, month_ends

__all__ = ["MomentumSpec", "Signal", "compute_signal", "month_ends"]
