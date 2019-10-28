from os import path
import sys

module_dir = path.abspath(__path__[0])
if module_dir not in sys.path:  # Check if the path is already on the search path.
    sys.path.append(module_dir)