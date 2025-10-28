"""DRIFT: drift-bounded distillation of one-step video diffusion into integer recurrent students."""
from .contraction import contraction_penalty, drift_bound, jacobian_spectral_norm
from .losses import consistency_matching, distillation_loss, residual_suppression
from .metrics import drift_at, state_error_trajectory
from .model import DriftStudent, warp
from .quant import LSQ, QConv2d, set_quant
from .statebank import StateBank

__version__ = "0.1.0"
__all__ = ["DriftStudent", "warp", "LSQ", "QConv2d", "set_quant", "StateBank", "jacobian_spectral_norm",
           "contraction_penalty", "drift_bound", "consistency_matching", "residual_suppression",
           "distillation_loss", "drift_at", "state_error_trajectory"]
