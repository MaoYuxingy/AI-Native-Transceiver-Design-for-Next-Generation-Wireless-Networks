"""Build the journal-style A1 poster (poster_A1_v4.pdf).

Figures, the logo and the QR code are inlined, so the page is self-contained and
headless Chrome needs no file access beyond the HTML itself.

    python build_jr.py
    chrome --headless --no-pdf-header-footer --print-to-pdf=... poster_v4.html
"""
import base64
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir)

# inlined verbatim: these are already vector
INLINE_SVG = (('__SYSFIG__', 'sysfig.svg'), ('__QR__', 'qr_repo.svg'))

# base64-embedded rasters
IMAGES = (('__IMG_AWGN__', 'awgn_ae_vs_qam_poster.png'),
          ('__IMG_RL__', 'rl_vs_gradient_awgn.png'),
          ('__LOGO_BLACK__', 'uol_logo_black.png'))

css = io.open(os.path.join(HERE, 'jr1.html'), encoding='utf-8').read()
body = (io.open(os.path.join(HERE, 'jr2.html'), encoding='utf-8').read()
        + io.open(os.path.join(HERE, 'jr3.html'), encoding='utf-8').read())

for token, name in INLINE_SVG:
    body = body.replace(token, io.open(os.path.join(HERE, name), encoding='utf-8').read())
    assert token not in body, token

for token, name in IMAGES:
    with open(os.path.join(SRC, name), 'rb') as fh:
        body = body.replace(token, 'data:image/png;base64,'
                            + base64.b64encode(fh.read()).decode())
    assert token not in body, token

doc = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
       '<title>LivSURF 436026 poster</title>\n'
       + css + '</head><body>\n' + body + '</body></html>\n')

out = os.path.join(HERE, 'poster_v4.html')
io.open(out, 'w', encoding='utf-8', newline='\n').write(doc)
print('poster_v4.html  %.2f MB' % (os.path.getsize(out) / 1024 / 1024))
