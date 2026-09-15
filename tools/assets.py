#!/usr/bin/env python3
"""Collect and web-optimize every figure referenced by supplementary.tex."""
import re, os, json, subprocess, shutil

PAPER = os.environ.get('CC_PAPER_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get('CC_SITE_DIR', os.path.dirname(OUT))
IMGDIR = os.path.join(SITE, 'static', 'images')
os.makedirs(IMGDIR, exist_ok=True)

# Figures the page shows itself, beyond whatever the supplement references.
EXTRA = ['figures/teaser.pdf', 'figures/pipeline.pdf', 'figures/benchmark_design.pdf',
         'figures/benchmark_stats.pdf']

# Figures the ICRA paper supersedes: web stem -> (source relative to PAPER, crop applied
# in the paper via \includegraphics trim). Keeps the page's headline figure in step with
# the submitted PDF.
OVERRIDE = {}

def resolve(p):
    if p in OVERRIDE:
        f = os.path.join(PAPER, OVERRIDE[p][0])
        if os.path.exists(f):
            return f
    cands = [p, p + '.pdf', p + '.png', p + '.jpg', p + '.jpeg']
    for c in cands:
        f = os.path.join(PAPER, c)
        if os.path.exists(f):
            return f
    return None

DPI = 160
CM = DPI / 2.54          # pixels per cm at the render resolution

def crop_args(trim, dst):
    """LaTeX trim is 'left bottom right top'; translate it to an ImageMagick crop."""
    l, b, r, t = (float(v.rstrip('cm')) * CM for v in trim.split())
    w, h = subprocess.run(['identify', '-format', '%w %h', dst],
                          capture_output=True, text=True, check=True).stdout.split()
    w, h = int(w), int(h)
    return ['-crop', '%dx%d+%d+%d' % (round(w - l - r), round(h - t - b), round(l), round(t)),
            '+repage']

def convert(src, stem, trim=None):
    ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(IMGDIR, stem + '.jpg')
    if ext == '.pdf':
        dst = os.path.join(IMGDIR, stem + '.png')
        subprocess.run(['pdftoppm', '-png', '-r', str(DPI), '-singlefile', src,
                        os.path.join(IMGDIR, stem)], check=True)
        crop = crop_args(trim, dst) if trim else []
        subprocess.run(['convert', dst] + crop + ['-resize', '1600x1600>', '-strip', dst],
                       check=True)
    else:
        subprocess.run(['convert', src, '-resize', '1400x1400>', '-quality', '86',
                        '-strip', dst], check=True)
    return 'static/images/' + os.path.basename(dst)

def read_supplement():
    """supplementary.tex plus every supp/*.tex it inputs."""
    top = open(os.path.join(PAPER, 'supplementary.tex')).read()
    parts = [top]
    for child in re.findall(r'\\input\{([^}]*)\}', top):
        f = os.path.join(PAPER, child if child.endswith('.tex') else child + '.tex')
        if os.path.exists(f):
            parts.append(open(f).read())
    return '\n'.join(parts)

def main():
    tex = read_supplement()
    tex = '\n'.join(re.sub(r'(?<!\\)%.*', '', l) for l in tex.split('\n'))
    tex = re.sub(r'\\begin\{comment\}.*?\\end\{comment\}', '', tex, flags=re.S)
    paths = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', tex)
    mapping, missing = {}, []
    for p in dict.fromkeys(paths + EXTRA):
        src = resolve(p)
        if not src:
            missing.append(p); continue
        stem = re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(p)[0]).strip('_')
        try:
            mapping[p] = convert(src, stem, OVERRIDE[p][1] if p in OVERRIDE else None)
        except subprocess.CalledProcessError as e:
            missing.append(p)
    json.dump(mapping, open(os.path.join(OUT, 'assets.json'), 'w'), indent=1)
    print('converted', len(mapping), 'missing', missing)

if __name__ == '__main__':
    main()
