"""Find module-level rebinding of the notebook's global constants.

A notebook shares one namespace across every cell, so a throwaway loop variable can
silently overwrite a constant that a class defined ten cells earlier depends on.
Comprehension targets are function-scoped in Python 3 and cannot leak; for-loops,
assignments and with/except targets at module level can."""
import io, json, ast, sys
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                  'Autoencoder_YuxingMao.ipynb')
nb = json.load(io.open(NB, encoding='utf-8'))
cells = nb['cells']

params = next(c for c in cells if c['id'] == 'eb0ccb86')
tree = ast.parse(''.join(params['source']))
CONST = {t.id for node in tree.body if isinstance(node, ast.Assign)
         for t in node.targets if isinstance(t, ast.Name)}
print('constants defined in the parameters cell: %d' % len(CONST))
print('  ' + ', '.join(sorted(CONST)))
print()

def module_level_binds(node):
    """Names bound at module level, not descending into def/class/comprehensions."""
    out = []
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(stmt):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                out.append((sub.id, sub.lineno))
            elif isinstance(sub, ast.For):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        out.append((t.id, sub.lineno))
    return out

hits = []
for i, c in enumerate(cells):
    if c['cell_type'] != 'code' or c['id'] == 'eb0ccb86':
        continue
    src = '\n'.join('pass' if l.strip().startswith('%') else l
                    for l in ''.join(c['source']).split('\n'))
    try:
        t = ast.parse(src)
    except SyntaxError:
        continue
    for name, ln in module_level_binds(t):
        if name in CONST:
            hits.append((i, c['id'], name, ln, src.split('\n')[ln - 1].strip()[:70]))

if hits:
    print('SHADOWED CONSTANTS:')
    for i, cid, name, ln, line in hits:
        print('  cell %-3d %-18s rebinds %-28s line %d: %s' % (i, cid, name, ln, line))
    sys.exit(1)
print('no constant is rebound at module level')
