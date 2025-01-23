"""
l2g-light: Infer global embeddings from local graph embeddings trained in parallel
"""

from .utils import DataLoader
from .embedding.gae.models import GAE, VGAE

__all__ = [
    "DataLoader",
    "GAE", 
    "VGAE"
]

