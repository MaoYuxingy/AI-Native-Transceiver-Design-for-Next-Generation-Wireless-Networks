import base64, io, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
imgs = {'__IMG_GLOBAL__': 'global_comparison_poster.png',
        '__IMG_AWGN__': 'awgn_ae_vs_qam_poster.png',
        '__IMG_RL__': 'rl_vs_gradient_awgn.png',
        '__LOGO__': 'uol_logo_white.png'}
css = io.open('pn1.html', encoding='utf-8').read()
body = io.open('pn2.html', encoding='utf-8').read() + io.open('pn3.html', encoding='utf-8').read()
body = body.replace('__SYSFIG__', io.open('sysfig.svg', encoding='utf-8').read())
for tok, fn in imgs.items():
    with open(os.path.join(SRC, fn), 'rb') as fh:
        body = body.replace(tok, 'data:image/png;base64,' + base64.b64encode(fh.read()).decode())
    assert tok not in body
doc = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
       '<title>LivSURF 436026 poster</title>\n' + css + '</head><body>\n' + body + '</body></html>\n')
out = os.path.abspath('poster_v2.html')
io.open(out, 'w', encoding='utf-8', newline='\n').write(doc)
print('poster_v2.html  %.2f MB' % (os.path.getsize(out) / 1024 / 1024))
