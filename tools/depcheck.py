# Static "Run All" check: does every cell only use names bound by an earlier cell?
import io, json, ast, builtins
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                  'Autoencoder_YuxingMao.ipynb')
nb = json.load(io.open(NB, encoding='utf-8'))

def names(tree):
    loads, stores = set(), set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Name):
            (loads if isinstance(nd.ctx, ast.Load) else stores).add(nd.id)
        elif isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(nd.name)
            for a in getattr(nd, 'args', ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])).args:
                stores.add(a.arg)
            if getattr(nd, 'args', None):
                for a in nd.args.posonlyargs + nd.args.kwonlyargs:
                    stores.add(a.arg)
                if nd.args.vararg: stores.add(nd.args.vararg.arg)
                if nd.args.kwarg: stores.add(nd.args.kwarg.arg)
        elif isinstance(nd, (ast.Import, ast.ImportFrom)):
            for al in nd.names:
                stores.add((al.asname or al.name).split('.')[0])
        elif isinstance(nd, ast.ExceptHandler) and nd.name:
            stores.add(nd.name)
        elif isinstance(nd, (ast.comprehension,)):
            pass
        elif isinstance(nd, ast.Global):
            stores.update(nd.names)
    return loads, stores

known = set(dir(builtins)) | {'__name__', 'get_ipython'}
problems = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    src = '\n'.join('pass' if l.strip().startswith('%') else l for l in src.split('\n'))
    loads, stores = names(ast.parse(src))
    missing = sorted(loads - stores - known)
    if missing:
        problems.append((i, c['id'], missing))
    known |= stores

if problems:
    print("POSSIBLY UNDEFINED AT RUN-ALL TIME:")
    for i, cid, m in problems:
        print(f"  cell {i} ({cid}): {m}")
else:
    print("Run-All dependency check: clean - every name is bound by an earlier cell.")
