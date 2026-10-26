from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .sfm_solver import SFMDetSolver
TASKS = {"classification": ClasSolver, "detection": DetSolver, "sfm_detection": SFMDetSolver}
