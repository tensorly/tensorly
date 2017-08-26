from slimit import minify
from rcssmin import cssmin

from pathlib import Path
asset_path = Path('./themes/tensorly/static')

for path in asset_path.glob('*.js'):
    # Ignore already minified files
    if '.min.' in str(path):
        continue
    target_path = path.with_suffix('.min.js')
    with open(path.as_posix(), 'r') as f:
        text = f.read()
    minified = minify(text, mangle=True, mangle_toplevel=True)
    with open(target_path.as_posix(), 'w') as f:
        f.write(minified)

for path in asset_path.glob('*.css'):
    # Ignore already minified files
    if '.min.' in str(path):
        continue
    target_path = path.with_suffix('.min.css')
    with open(path.as_posix(), 'r') as f:
        text = f.read()
    minified = cssmin(text)
    with open(target_path.as_posix(), 'w') as f:
        f.write(minified)

