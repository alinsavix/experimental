import pathlib
import sys

# Make `import ubxread` work no matter where pytest is invoked from.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
