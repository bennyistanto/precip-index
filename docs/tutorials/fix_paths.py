import re
from pathlib import Path

# Setup code to insert
setup_code = '''```{python}
import sys
import os
from pathlib import Path

# Get the notebook directory and find the repository root
notebook_dir = Path.cwd()
# Go up to repo root (from docs/tutorials/ or wherever we are)
repo_root = notebook_dir
while not (repo_root / 'src').exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

# Add src to path
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt'''

for file in ['02-calculate-spei.qmd', '03-event-analysis.qmd', '04-visualization.qmd']:
    content = Path(file).read_text(encoding='utf-8')
    
    # Replace the old path setup pattern
    pattern = r'```\{python\}\nimport sys\nimport os\n\n# Add src directory.*?\nsys\.path\.insert\(0, os\.path\.abspath\(.*?\)\)\n\nimport numpy as np\nimport xarray as xr\nimport matplotlib\.pyplot as plt'
    
    # Find and replace
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, setup_code, content, count=1, flags=re.DOTALL)
        Path(file).write_text(content, encoding='utf-8')
        print(f"Fixed {file}")
    else:
        print(f"Pattern not found in {file}")
