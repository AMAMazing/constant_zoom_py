import sys
from .main import smartzoom

# This allows the user to call the function directly after `import smartzoom`,
# instead of having to do `smartzoom.smartzoom`.
sys.modules[__name__] = smartzoom
