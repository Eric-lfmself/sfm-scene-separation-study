"""
Render the two figures as standalone SVG, in a light and a dark variant.

GitHub does not execute JavaScript in a repository, so the figures are emitted as
static SVG with literal colours rather than drawn at view time. Markdown then picks
the variant with a <picture> element and prefers-color-scheme.

    python tools/make_figures.py        # writes figures/*.svg

Palette: the two series colours were checked for colour-vision separation against
their own surface (adjacent-pair Delta E ~21-23 under protanopia/deuteranopia, well
above the 8 floor), and each sits inside its mode's lightness band.
"""

import math
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

THEMES = {
    'light': dict(bg='#ffffff', ink='#141a21', ink2='#48545f', muted='#77848f',
                  rule='#dbe2e9', rule_strong='#c3ccd6',
                  within='#0f6fb5', cross='#c1622c'),
    'dark':  dict(bg='#0d1117', ink='#e7edf3', ink2='#a5b2be', muted='#7d8a95',
                  rule='#28303a', rule_strong='#39434f',
                  within='#4f9bd8', cross='#d1712f'),
}

SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def text(x, y, s, fill, size=11, family=MONO, anchor='start', weight='normal'):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-family="{family}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{esc(s)}</text>')


# --------------------------------------------------------------- figure 1

EDGES = [15, 22, 32, 46, 68, 100, 146, 215, 316, 464, 681, 1000, 1468, 2154, 3400]
WITHIN = [39, 50, 47, 48, 54, 64, 75, 80, 107, 92, 94, 55, 47, 38]
CROSS = [71, 38, 19, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def figure_inlier_histogram(t):
    W, H = 880, 360
    ml, mr, mt, mb = 46, 14, 58, 56
    pw, ph = W - ml - mr, H - mt - mb
    n = len(WITHIN)
    y_max = 120
    gw = pw / n
    bw = min(19.0, (gw - 6) / 2)
    gap = 2.0

    def y(v):
        return mt + ph - (v / y_max) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="Histogram of RANSAC inliers per verified '
         f'image pair. Cross-scene pairs are confined below 83 inliers; within-scene '
         f'pairs spread to 3379.">',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>']

    # title + legend
    p.append(text(0, 18, 'Inliers per verified image pair', t['ink'], 14, SANS, weight='600'))
    lx = 0
    for label, col in (('Within-scene (n=890)', t['within']), ('Cross-scene (n=150)', t['cross'])):
        p.append(f'<rect x="{lx}" y="30" width="11" height="11" rx="2" fill="{col}"/>')
        p.append(text(lx + 17, 40, label, t['ink2'], 12, SANS))
        lx += 17 + len(label) * 6.6 + 22

    # gridlines
    for v in range(0, y_max + 1, 30):
        p.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                 f'stroke="{t["rule"]}" stroke-width="1" shape-rendering="crispEdges"/>')
        p.append(text(ml - 10, y(v) + 4, v, t['muted'], 11, MONO, 'end'))
    p.append(text(ml - 10, mt - 14, 'pairs', t['muted'], 10.5, MONO, 'end'))

    # threshold marker at 100 inliers, which is a bin boundary
    tx = ml + 5 * gw
    p.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{mt-4}" y2="{mt+ph}" '
             f'stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    p.append(text(tx - 9, mt + 20, '100-inlier filter', t['ink2'], 11.5, MONO, 'end', '500'))
    p.append(text(tx - 9, mt + 36, 'all of this is dropped', t['muted'], 11, MONO, 'end'))

    # bars
    for i in range(n):
        cx = ml + i * gw + gw / 2
        for val, col, x0 in ((WITHIN[i], t['within'], cx - bw - gap / 2),
                             (CROSS[i], t['cross'], cx + gap / 2)):
            if val == 0:
                continue
            h = max(2.0, (val / y_max) * ph)
            yy = mt + ph - h
            r = min(3.0, h / 2)
            p.append(f'<path d="M{x0:.1f},{yy+h:.1f} L{x0:.1f},{yy+r:.1f} '
                     f'Q{x0:.1f},{yy:.1f} {x0+r:.1f},{yy:.1f} L{x0+bw-r:.1f},{yy:.1f} '
                     f'Q{x0+bw:.1f},{yy:.1f} {x0+bw:.1f},{yy+r:.1f} '
                     f'L{x0+bw:.1f},{yy+h:.1f} Z" fill="{col}"/>')

    # x axis
    p.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{mt+ph}" y2="{mt+ph}" '
             f'stroke="{t["rule_strong"]}" stroke-width="1" shape-rendering="crispEdges"/>')
    for i in (0, 2, 4, 5, 7, 9, 11, 14):
        x = ml + i * gw
        p.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt+ph}" y2="{mt+ph+5}" '
                 f'stroke="{t["rule_strong"]}" stroke-width="1"/>')
        p.append(text(x, mt + ph + 19, EDGES[i], t['muted'], 11, MONO, 'middle'))
    p.append(text(ml + pw / 2, H - 10, 'RANSAC inliers per verified pair (log-spaced bins)',
                  t['muted'], 12, SANS, 'middle'))
    p.append('</svg>')
    return '\n'.join(p)


# --------------------------------------------------------------- figure 2

ROWS = [
    dict(label='Within-scene',     sub='true - same scene',    n=890, mn=15, p10=32, med=302, p90=1421, mx=3379, keep=True),
    dict(label='Cross-session',    sub='true - revisit',       n=370, mn=15, p10=18, med=51,  p90=225,  mx=860,  keep=True),
    dict(label='Cross-scene',      sub='FALSE - lookalikes',   n=150, mn=15, p10=16, med=22,  p90=54,   mx=83,   keep=False),
    dict(label='Cross-site (UAV)', sub='FALSE - unlike sites', n=195, mn=15, p10=15, med=16,  p90=19,   mx=37,   keep=False),
]


def figure_link_ranges(t):
    W, H = 880, 322
    ml, mr, mt, mb = 222, 24, 54, 52
    pw, ph = W - ml - mr, H - mt - mb
    lo, hi = math.log10(12), math.log10(4000)

    def x(v):
        return ml + (math.log10(v) - lo) / (hi - lo) * pw

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="Inlier ranges for four kinds of image-pair '
         f'link on a log scale. True within-scene links have a median of 302 and true '
         f'cross-session revisit links a median of 51. False cross-scene links between '
         f'lookalike places have a median of 22 and reach 83, overlapping the true revisit '
         f'population. False cross-site links between visually unlike aerial sites have a '
         f'median of 16 and never exceed 37.">',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>']

    p.append(text(0, 18, 'Where each kind of link lives', t['ink'], 14, SANS, weight='600'))
    lx = 0
    for label, col in (('Should be kept', t['within']), ('Should be dropped', t['cross'])):
        p.append(f'<rect x="{lx}" y="30" width="11" height="11" rx="2" fill="{col}"/>')
        p.append(text(lx + 17, 40, label, t['ink2'], 12, SANS))
        lx += 17 + len(label) * 6.6 + 22

    for v in (20, 50, 100, 200, 500, 1000, 2000, 4000):
        p.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{mt-6}" y2="{mt+ph}" '
                 f'stroke="{t["rule"]}" stroke-width="1" shape-rendering="crispEdges"/>')
        p.append(text(x(v), mt + ph + 18, v, t['muted'], 11, MONO, 'middle'))

    p.append(f'<line x1="{x(100):.1f}" x2="{x(100):.1f}" y1="{mt-10}" y2="{mt+ph}" '
             f'stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    p.append(text(x(100), mt - 15, '100-inlier filter', t['ink2'], 11.5, MONO, 'middle', '500'))

    step = ph / len(ROWS)
    for i, r in enumerate(ROWS):
        yy = mt + step * (i + 0.5)
        col = t['within'] if r['keep'] else t['cross']
        p.append(text(ml - 16, yy - 1, r['label'], t['ink'], 13, SANS, 'end', '500'))
        p.append(text(ml - 16, yy + 15, f"{r['sub']} - n={r['n']}", t['muted'], 11, MONO, 'end'))
        p.append(f'<line x1="{x(r["mn"]):.1f}" x2="{x(r["mx"]):.1f}" y1="{yy:.1f}" '
                 f'y2="{yy:.1f}" stroke="{col}" stroke-width="2" opacity="0.45"/>')
        for v in (r['mn'], r['mx']):
            p.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{yy-6:.1f}" y2="{yy+6:.1f}" '
                     f'stroke="{col}" stroke-width="2" opacity="0.45"/>')
        p.append(f'<rect x="{x(r["p10"]):.1f}" y="{yy-9:.1f}" '
                 f'width="{max(2.0, x(r["p90"])-x(r["p10"])):.1f}" height="18" rx="3" '
                 f'fill="{col}" opacity="0.85"/>')
        p.append(f'<circle cx="{x(r["med"]):.1f}" cy="{yy:.1f}" r="5" fill="{t["bg"]}" '
                 f'stroke="{col}" stroke-width="2.5"/>')

    p.append(text(ml + pw / 2, H - 8, 'RANSAC inliers per verified pair (log scale)',
                  t['muted'], 12, SANS, 'middle'))
    p.append('</svg>')
    return '\n'.join(p)


def main():
    os.makedirs(OUT, exist_ok=True)
    for mode, t in THEMES.items():
        for name, fn in (('inlier-histogram', figure_inlier_histogram),
                         ('link-ranges', figure_link_ranges)):
            path = os.path.join(OUT, f'{name}-{mode}.svg')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(fn(t))
            print('wrote', os.path.relpath(path, os.path.dirname(OUT)))


if __name__ == '__main__':
    main()
