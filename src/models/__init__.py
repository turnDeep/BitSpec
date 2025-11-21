# src/models/__init__.py
"""
Model Definition Module
"""

# from .gcn_model import GCNMassSpecPredictor
# from .graph_transformer import GraphTransformerPredictor
# from .baseline import BaselinePredictor
from .teacher import TeacherModel
from .student import StudentModel

__all__ = [
    # "GCNMassSpecPredictor",
    "TeacherModel",
    "StudentModel"
]
