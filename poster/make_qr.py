"""Regenerate poster/qr_repo.svg, the QR code in the poster header.

Run this only when the repository URL changes; build_pn.py inlines the committed
SVG, so a normal poster rebuild does not need segno installed.

    pip install segno && python make_qr.py

Dark modules on white, never reversed out of the navy header: many scanners fail on
inverted symbols, so the code keeps its own white tile and quiet zone.
"""
import io
import re

import segno

URL = ('https://github.com/MaoYuxingy/'
       'AI-Native-Transceiver-Design-for-Next-Generation-Wireless-Networks')

qr = segno.make(URL, error='m')
buf = io.BytesIO()
qr.save(buf, kind='svg', scale=1, border=2, dark='#0D2340', light='#ffffff',
        xmldecl=False, svgns=True, nl=False)
svg = buf.getvalue().decode('utf-8')

# Swap the fixed width/height for a viewBox so the poster can size it in millimetres.
m = re.search(r'width="([\d.]+)" height="([\d.]+)"', svg)
svg = svg.replace(m.group(0),
                  'viewBox="0 0 {0} {0}" width="100%" height="100%"'.format(m.group(1)), 1)
io.open('qr_repo.svg', 'w', encoding='utf-8', newline='\n').write(svg)

across = qr.symbol_size(scale=1, border=2)[0]
print('version %s-%s, %d modules across including the quiet zone' % (qr.version, qr.error.upper(), across))
print('printed 34 mm wide that is %.2f mm per module' % (34.0 / across))
