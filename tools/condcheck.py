"""Names the notebook relies on that are NOT bound on every execution path.

depcheck.py treats any binding anywhere in a cell as available to later cells. That is
wrong for a binding inside `except ImportError:` or one arm of an `if`: the branch may
never run, and the name is then missing at run time with nothing static to show for it.
`import os` sat in an `except ImportError:` handler for exactly this reason and only
surfaced as a NameError mid-run.

A name is reported when all three hold:
  * cell A binds it, but not on every path through A
  * some later cell B loads it
  * B does not bind it itself (which rules out loop variables and locals)
"""
import json, io, ast, sys, collections
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                  'Autoencoder_YuxingMao.ipynb')
NL = chr(10)
nb = json.load(io.open(NB, encoding='utf-8'))

def src(c):
    return NL.join(l for l in ''.join(c['source']).split(NL)
                   if not l.lstrip().startswith(('%', '!')))

def binds(node):
    """Every name this node binds, ignoring path conditions."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split('.')[0] for a in n.names)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                out.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            out.update(x.id for x in ast.walk(n.target) if isinstance(x, ast.Name))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            out.update(x.id for x in ast.walk(n.target) if isinstance(x, ast.Name))
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            out.update(x.id for x in ast.walk(n.optional_vars) if isinstance(x, ast.Name))
    return out

def certain(body):
    """Names bound on EVERY path through this statement list."""
    sure = set()
    for st in body:
        if isinstance(st, ast.If):
            # only counts if both arms bind it
            sure |= certain(st.body) & certain(st.orelse) if st.orelse else set()
        elif isinstance(st, ast.Try):
            # the body may raise part-way, so only what every handler also binds is sure
            hs = [certain(h.body) for h in st.handlers]
            inner = certain(st.body)
            for h in hs:
                inner &= h
            sure |= inner | certain(st.finalbody)
        elif isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
            sure |= certain(st.orelse) if st.orelse else set()   # body may not run
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            sure |= binds(st)          # a with-body runs unconditionally
        else:
            sure |= binds(st)
    return sure

maybe, sure_all, loads, binds_in = {}, set(), collections.defaultdict(list), {}
for pos, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    try: t = ast.parse(src(c))
    except SyntaxError: continue
    b, s = binds(t), certain(t.body)
    binds_in[pos] = b
    for nm in b - s:
        maybe.setdefault(nm, pos)
    sure_all |= s
    for n in ast.walk(t):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads[n.id].append(pos)

bad = []
for nm, at in maybe.items():
    if nm in sure_all: continue                       # also bound unconditionally somewhere
    users = [p for p in loads.get(nm, []) if p > at and nm not in binds_in.get(p, ())]
    if users:
        bad.append((nm, at, sorted(set(users))))

for nm, at, users in sorted(bad):
    print('RISKY  %-16s not bound on every path through pos %d; loaded at %s'
          % (nm, at, users[:10]))
print('conditional-binding check: %s' % ('clean' if not bad else '%d issue(s)' % len(bad)))
sys.exit(1 if bad else 0)
