# cage_fusion/__init__.py

__version__ = "0.1.0"

# Expose the core API function at the top level of the package
from .api.predict import predict_smiles

# The imports below can remain for internal use or for advanced users
from . import configs
from . import models
from . import featurizers
from . import engine

# This defines the public API of your package. When a user types
# 'from cage_fusion import *', only 'predict_smiles' will be imported.
__all__ = ["predict_smiles"]