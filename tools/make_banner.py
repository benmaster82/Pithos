"""Generates docs/readme-banner.svg: the white Pandora mark on a line, with the wordmark as paths
(no fonts needed where GitHub renders it). Usage: python tools/make_banner.py docs"""
import re, sys
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

D = sys.argv[1]  # the docs folder

def text_path(font_file, text, size, x, y, spacing=0.0):
    f = TTFont(font_file)
    gs, cmap, hmtx = f.getGlyphSet(), f.getBestCmap(), f['hmtx']
    s = size / f['head'].unitsPerEm
    out, cx = [], x
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, cx, y)))
        out.append(pen.getCommands())
        cx += hmtx[g][0] * s + spacing
    return ' '.join(out), cx - spacing

def width(font_file, text, size, spacing=0.0):
    return text_path(font_file, text, size, 0, 0, spacing)[1]

mark = open(f'{D}/assets/pandora-mark.svg', encoding='utf-8').read()
mark_d = re.search(r' d="([^"]+)"', mark).group(1)

W, H = 1280, 400
LINE_Y = 262
M = 168                      # mascot size (its 100-unit box)
MX, MY = 132, LINE_Y - 0.918 * M   # the mark's lowest point is at about y = 91.8 of 100
fonts = f'{D}/assets/fonts'
word, word_end = text_path(f'{fonts}/space-grotesk-bold.ttf', 'pithos', 132, 360, 222, spacing=-2)
sub = 'local inference engine · C11 · no dependencies'
sub_d, _ = text_path(f'{fonts}/ibm-plex-mono.ttf', sub, 19, 364, 310)
tag = 'LFM2 · LFM2.5 · LFM3'
tw = width(f'{fonts}/ibm-plex-mono.ttf', tag, 17)
tag_d, _ = text_path(f'{fonts}/ibm-plex-mono.ttf', tag, 17, W - 96 - tw, 310)
trail_end = MX + M * 0.5

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Pithos: local inference engine">
  <defs>
    <linearGradient id="trail" gradientUnits="userSpaceOnUse" x1="40" y1="0" x2="{trail_end:.1f}" y2="0"><stop offset="0" stop-color="#7c5cf6" stop-opacity="0"/><stop offset=".35" stop-color="#7c5cf6"/><stop offset=".7" stop-color="#ec5c8c"/><stop offset="1" stop-color="#ffc460"/></linearGradient>
    <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#ffffff" fill-opacity=".045"/></pattern>
  </defs>
  <rect width="{W}" height="{H}" rx="22" fill="#08090b"/>
  <rect width="{W}" height="{H}" rx="22" fill="url(#dots)"/>
  <rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="21.5" fill="none" stroke="#ffffff" stroke-opacity=".08"/>
  <line x1="96" y1="{LINE_Y}" x2="{W-96}" y2="{LINE_Y}" stroke="#ffffff" stroke-opacity=".14"/>
  <line x1="40" y1="{LINE_Y}" x2="{trail_end:.1f}" y2="{LINE_Y}" stroke="url(#trail)" stroke-width="2">
    <animate attributeName="x2" from="40" to="{trail_end:.1f}" dur="1.4s" calcMode="spline" keySplines=".2 .8 .2 1" fill="freeze"/>
  </line>
  <g transform="translate({MX} {MY:.1f}) scale({M/100})">
    <path fill="#ffffff" fill-rule="evenodd" d="{mark_d}"/>
  </g>
  <path fill="#ffffff" d="{word}"/>
  <path fill="#8b949e" d="{sub_d}"/>
  <path fill="#5c646d" d="{tag_d}"/>
</svg>
'''
open(f'{D}/readme-banner.svg', 'w', encoding='utf-8', newline='\n').write(svg)
print('ok', len(svg), 'bytes; wordmark ends at', round(word_end))
