"""Embed text resources into C headers as NUL-terminated byte arrays (outputs are committed):
    src/kernels.cl   -> src/kernels_cl.h  (KERNELS_CL)
    web/index.html   -> src/webui_html.h  (WEBUI_HTML)
Run after editing either file:  python tools/embed.py"""
import os

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.normpath(os.path.join(here, '..'))
items = [('src/kernels.cl', 'src/kernels_cl.h', 'KERNELS_CL', 'char'),
         ('web/index.html', 'src/webui_html.h', 'WEBUI_HTML', 'unsigned char')]
for src, dst, name, ctype in items:
    data = open(os.path.join(root, src), 'rb').read().replace(b'\r\n', b'\n')
    lines = ['/* generated from %s by tools/embed.py - do not edit */' % src,
             'static const %s %s[] = {' % (ctype, name)]
    payload = list(data) + [0]
    for i in range(0, len(payload), 20):
        lines.append('    ' + ', '.join('0x%02x' % b for b in payload[i:i + 20]) + ',')
    lines.append('};')
    open(os.path.join(root, dst), 'w', newline='\n').write('\n'.join(lines) + '\n')
    print('%s: %d bytes' % (dst, len(data)))
