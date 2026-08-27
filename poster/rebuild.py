"""Rebuild a poster PDF from the current figures, in one command.

    python rebuild.py            # journal design  -> poster_A1_v4.pdf
    python rebuild.py card       # card design     -> poster_A1_v3.pdf

Run this after re-running the notebook, so the poster picks up the regenerated
figures. It inlines everything, drives headless Chrome, and then checks the result
is a single A1 page - a second page means the content outgrew the sheet.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

DESIGNS = {
    'journal': ('build_jr.py', 'poster_v4.html', 'poster_A1_v4.pdf'),
    'card': ('build_pn.py', 'poster_v2.html', 'poster_A1_v3.pdf'),
}

CHROME_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
]


def find_chrome():
    for path in CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    sys.exit('No Chrome or Edge found. Add its path to CHROME_CANDIDATES.')


def page_count(pdf):
    """Pages in the PDF, or None if pdfinfo is not on PATH."""
    try:
        out = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    for line in out.splitlines():
        if line.startswith('Pages:'):
            return int(line.split(':')[1])
    return None


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else 'journal').lower()
    if which not in DESIGNS:
        sys.exit('Unknown design %r. Choose from: %s' % (which, ', '.join(DESIGNS)))
    builder, html, pdf_name = DESIGNS[which]

    subprocess.run([sys.executable, os.path.join(HERE, builder)], check=True, cwd=HERE)

    pdf = os.path.join(ROOT, pdf_name)
    subprocess.run([find_chrome(), '--headless', '--disable-gpu',
                    '--no-pdf-header-footer', '--print-to-pdf=' + pdf,
                    os.path.join(HERE, html)], check=True)

    size = os.path.getsize(pdf) / 1024 / 1024
    pages = page_count(pdf)
    print('%s  %.2f MB' % (pdf_name, size))
    if pages == 1:
        print('one A1 page - good')
    elif pages is None:
        print('pdfinfo not on PATH; check the page count by opening the file')
    else:
        print('WARNING: %d pages. The content no longer fits; trim a section '
              'or shrink a figure.' % pages)


if __name__ == '__main__':
    main()
