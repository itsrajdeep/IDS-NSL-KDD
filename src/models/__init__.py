"""
__init__.py — Dynamic Model Registry for the IDS project.

FROZEN: Do not modify after initial setup.

WHY THIS FILE EXISTS:
    Without this, the benchmark runner (evaluator.py) would have to do:

        from src.models.tree_models import DecisionTreeModel, RandomForestModel, ...
        from src.models.boosting_models import LightGBMModel, XGBoostModel, ...
        models = [DecisionTreeModel(), RandomForestModel(), LightGBMModel(), ...]

    That means evaluator.py must be updated every time a model is added.
    Worse, it creates a merge conflict — both devs edit the same runner file.

    WITH THIS REGISTRY:
        from src.models import MODEL_REGISTRY
        models = [cls() for cls in MODEL_REGISTRY.values()]

    The runner never changes. Each dev just adds a class to their own file.
    Git auto-merges. Zero conflicts.

HOW THE REGISTRY WORKS:
    1. This file tries to import everything from tree_models.py and boosting_models.py.
    2. It then scans all imported names for classes that are subclasses of BaseIDSModel.
    3. It stores them in MODEL_REGISTRY = { "LightGBM": LightGBMModel, ... }
    4. Anyone who needs all models does: from src.models import MODEL_REGISTRY
"""

# The registry dictionary: { model_name_string : model_class }
# e.g. { "LightGBM": LightGBMModel, "DecisionTree": DecisionTreeModel, ... }
MODEL_REGISTRY = {}

# -----------------------------------------------------------------
# STEP 1: Import all model classes from both dev files
# -----------------------------------------------------------------
# 'try/except ImportError' means:
#   "If tree_models.py doesn't exist yet (4kub0 hasn't pushed it),
#    don't crash — just skip it. The registry will only have what exists."
#
# This lets both devs work independently. If you push boosting_models.py
# and 4kub0 hasn't pushed tree_models.py yet, the system still works
# with just your models (and vice versa).

try:
    # 'from ... import *' imports ALL public names from that module
    # (i.e., all classes that don't start with an underscore)
    from src.models.tree_models import *       # noqa: F401, F403
except (ImportError, ModuleNotFoundError):
    # tree_models.py doesn't exist yet — that's fine, skip it silently
    pass

try:
    from src.models.boosting_models import *   # noqa: F401, F403
except (ImportError, ModuleNotFoundError):
    # boosting_models.py doesn't exist yet — skip it silently
    pass

# -----------------------------------------------------------------
# STEP 2: Auto-discover all BaseIDSModel subclasses in this namespace
# -----------------------------------------------------------------
import inspect   # inspect.isclass() checks if something is a class
import sys       # sys.modules lets us look at the current module's namespace

from src.contracts import BaseIDSModel

# sys.modules[__name__] = this module itself (__init__.py)
# vars(...) returns its entire namespace as a dict { name: object }
_current_module = sys.modules[__name__]

for _attr_name, _obj in list(vars(_current_module).items()):

    # inspect.isclass(_obj)            → Is this a class (not a function, int, etc.)?
    # issubclass(_obj, BaseIDSModel)   → Does it inherit from our contract?
    # _obj is not BaseIDSModel         → Exclude the base class itself
    # _obj is not IDSModelMixin        → Exclude the mixin itself (if imported)
    # _obj.name                        → Has a non-empty name string set
    if (
        inspect.isclass(_obj)
        and issubclass(_obj, BaseIDSModel)
        and _obj is not BaseIDSModel
        and isinstance(getattr(_obj, 'name', None), str)
        and getattr(_obj, 'name', '') != ''
    ):
        # Register: "LightGBM" → LightGBMModel class
        MODEL_REGISTRY[_obj.name] = _obj


# -----------------------------------------------------------------
# CONVENIENCE: __all__ tells 'from src.models import *' what to expose
# -----------------------------------------------------------------
# Only MODEL_REGISTRY is meant for external use.
# Internal helpers (_current_module, _obj, etc.) stay private.
__all__ = ['MODEL_REGISTRY']
