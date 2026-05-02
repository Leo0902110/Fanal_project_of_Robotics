from .ambiguity import VisualAmbiguityDetector
from .pseudo_blur import PseudoBlurConfig, apply_pseudo_blur

VisualUncertaintyDetector = VisualAmbiguityDetector

__all__ = ["PseudoBlurConfig", "VisualAmbiguityDetector", "VisualUncertaintyDetector", "apply_pseudo_blur"]
