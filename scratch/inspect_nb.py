import json

with open(r'c:/Users/shaha/Downloads/backend.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    header = lines[0] if lines else 'EMPTY'
    if len(lines) > 1 and lines[0].startswith('#'):
        header += ' | ' + lines[1]
    print(f"--- Cell {i} ({c.get('cell_type')}) ---")
    print(header[:100])
