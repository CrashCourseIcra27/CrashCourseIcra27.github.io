#!/usr/bin/env python3
"""Convert supplementary.tex (and the supp/*.tex it inputs) into the page body."""
import unicodedata, re, os, sys, json, html

PAPER = os.environ.get('CC_PAPER_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- utilities
def strip_comments(t):
    t = '\n'.join(re.sub(r'(?<!\\)%.*', '', l) for l in t.split('\n'))
    return re.sub(r'\\begin\{comment\}.*?\\end\{comment\}', '', t, flags=re.S)

def find_group(t, i):
    """t[i] must be '{'; return (content, index_after_closing_brace)."""
    assert t[i] == '{', t[i:i+30]
    d, j = 1, i + 1
    while d:
        if t[j] == '\\':
            j += 2; continue
        if t[j] == '{': d += 1
        elif t[j] == '}': d -= 1
        j += 1
    return t[i+1:j-1], j

def take_arg(t, m_end):
    """Read an optional [..] then a {..} argument starting at m_end."""
    j = m_end
    while j < len(t) and t[j] in ' \n': j += 1
    if j < len(t) and t[j] == '[':
        k = t.index(']', j); j = k + 1
        while j < len(t) and t[j] in ' \n': j += 1
    if j < len(t) and t[j] == '{':
        return find_group(t, j)
    return '', m_end

def env_span(t, name, start):
    """Return (inner, end_index_after_\\end{name}) for balanced environment."""
    b = '\\begin{%s}' % name
    e = '\\end{%s}' % name
    d, i = 1, start
    while d:
        nb = t.find(b, i); ne = t.find(e, i)
        if ne == -1: raise ValueError('unbalanced ' + name)
        if nb != -1 and nb < ne:
            d += 1; i = nb + len(b)
        else:
            d -= 1; i = ne + len(e)
    return t[start:i - len(e)], i

# ---------------------------------------------------------------- references
def load_bbl(path):
    txt = open(path).read()
    out = {}
    for m in re.finditer(r'\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})',
                         txt, flags=re.S):
        out[m.group(1)] = ' '.join(m.group(2).split())
    return out

BBL = load_bbl(os.path.join(PAPER, 'supplementary.bbl'))
BBL_MAIN = load_bbl(os.path.join(PAPER, 'main.bbl'))
for k, v in BBL_MAIN.items():
    BBL.setdefault(k, v)

CITE_ORDER = []          # keys in order of first appearance
def cite_num(key):
    if key not in CITE_ORDER:
        CITE_ORDER.append(key)
    return CITE_ORDER.index(key) + 1

# ------------------------------------------------------------- main.aux refs
def load_aux(path, prefix=''):
    out = {}
    if not os.path.exists(path): return out
    for line in open(path):
        m = re.match(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}', line)
        if m:
            out[prefix + m.group(1)] = m.group(2)
    return out

# Main-paper numbering comes from the ICRA submission, which is the paper this page
# accompanies. GREG_MAIN_AUX overrides it for a different build of the paper.
MAIN_AUX = os.environ.get('CC_MAIN_AUX', os.path.join(PAPER, 'main.aux'))
MAIN_LABELS = load_aux(MAIN_AUX, 'MAIN-')
assert MAIN_LABELS, 'no labels loaded from %s -- build the main paper first' % MAIN_AUX

LABELS = {}              # label -> (kind, display, anchor)

# ---------------------------------------------------------------- inline TeX
SIMPLE = [
    (r'\\MethodName\{\}', 'CrashCourse'), (r'\\MethodName\b', 'CrashCourse'),
    # Draft placeholders.  They are rendered, not silently dropped: an empty cell
    # would read as a measured blank instead of a missing experiment.
    (r'\\pcell\b', '<span class="pend">P</span>'),
    (r'\\boldparagraph\b', ''),
    (r'\\eg\b\.?', '<em>e.g</em>.'), (r'\\ie\b\.?', '<em>i.e</em>.'),
    (r'\\etal\b\.?', '<em>et al</em>.'), (r'\\etc\b\.?', '<em>etc</em>.'),
    (r'\\wrt\b', 'w.r.t.'), (r'\\vs\b', 'vs.'),
    (r'\\bluecheck\{\}', '<span class="yes">&#10003;</span>'),
    (r'\\bluecheck\b', '<span class="yes">&#10003;</span>'),
    (r'\\tikzcmark\b', '<span class="yes">&#10003;</span>'),
    (r'\\tikzxmark\b', '<span class="no">&#10007;</span>'),
    (r'\\cmark\b', '<span class="yes">&#10003;</span>'),
    (r'\\xmark\b', '<span class="no">&#10007;</span>'),
    (r'\\ding\{51\}', '<span class="yes">&#10003;</span>'),
    (r'\\ding\{55\}', '<span class="no">&#10007;</span>'),
    (r'\\noindent\b', ''), (r'\\centering\b', ''), (r'\\small\b', ''),
    (r'\\footnotesize\b', ''), (r'\\scriptsize\b', ''), (r'\\normalsize\b', ''),
    (r'\\bf\b', ''), (r'\\it\b', ''), (r'\\rm\b', ''),
    (r'\\ldots\b', '&hellip;'), (r'\\dots\b', '&hellip;'),
    (r'\\vspace\{[^}]*\}', ''), (r'\\hspace\{[^}]*\}', ''),
    (r'\\vskip[^\n]*', ''), (r'\\quad\b', ' '), (r'\\qquad\b', ' '),
    (r'\\!', ''), (r'\\,', ' '), (r'\\;', ' '),
    (r'\\linebreak\b', ' '), (r'\\newline\b', ' '), (r'\\par\b', ' '),
    (r'\\@', ''),
]

WRAPPERS = {
    'textbf': ('<strong>', '</strong>'),
    'textit': ('<em>', '</em>'),
    'emph': ('<em>', '</em>'),
    'texttt': ('<code>', '</code>'),
    'textsc': ('<span class="sc">', '</span>'),
    'underline': ('<u>', '</u>'),
    'mbox': ('', ''),
    'text': ('', ''),
    'hidden': ('', ''),
    'red': ('<span class="note">', '</span>'),
    'green': ('<span class="note">', '</span>'),
    'todo': ('<span class="note">TODO: ', '</span>'),
    'draftnote': ('<span class="note">', '</span>'),
    'pending': ('<span class="pend">[pending: ', ']</span>'),
    'apptab': ('<span class="sc">', '</span>'),
    'appsec': ('<span class="sc">', '</span>'),
}

MATH_RE = re.compile(r'(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])', re.S)

def protect_math(t, store):
    def rep(m):
        store.append(m.group(0))
        return '\x00%d\x00' % (len(store) - 1)
    return MATH_RE.sub(rep, t)

def restore_math(t, store):
    return re.sub(r'\x00(\d+)\x00', lambda m: html_escape_math(store[int(m.group(1))]), t)

def html_escape_math(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def ref_html(label):
    if label.startswith('eq:'):
        return '\\(\\ref{%s}\\)' % label
    if label.startswith('MAIN-'):
        num = MAIN_LABELS.get(label)
        if num: return '<span class="mainref">%s</span>' % num
        return '<span class="mainref">?</span>'
    if label in LABELS:
        kind, disp, anchor = LABELS[label]
        return '<a href="#%s">%s</a>' % (anchor, disp)
    # Table bodies are shared with the main paper, where a cross-document \ref
    # carries no MAIN- prefix (xr resolves it).  Fall back to the paper's own
    # number rather than emitting a dead link to a float this page never shows.
    num = MAIN_LABELS.get('MAIN-' + label)
    if num:
        return '<span class="mainref">%s</span>' % num
    return '<a href="#%s">%s</a>' % (label, '?')

# LaTeX accent commands -> precomposed Unicode. Bare forms are limited to the
# punctuation accents, which cannot be confused with a control word; the letter
# forms (c, v, u, H) must brace their argument for the same reason.
COMBINING = {'"': '\u0308', "'": '\u0301', '`': '\u0300', '^': '\u0302',
             '~': '\u0303', '=': '\u0304', '.': '\u0307',
             'c': '\u0327', 'v': '\u030c', 'u': '\u0306', 'H': '\u030b'}
ACCENT_RE = re.compile(r'\{?\\(?:(?P<p>["\'`^~=.])\s*\{?(?P<pl>[A-Za-z])\}?'
                       r'|(?P<w>[cvuH])(?:\{(?P<wl>[A-Za-z])\}|\s+(?P<wl2>[A-Za-z])))\}?')

def accents(t):
    def rep(m):
        a = m.group('p') or m.group('w')
        c = m.group('pl') or m.group('wl') or m.group('wl2')
        return unicodedata.normalize('NFC', c + COMBINING[a])
    return ACCENT_RE.sub(rep, t)


def inline(t, allow_cite=True):
    store = []
    t = protect_math(t, store)

    # citations
    def do_cite(m):
        keys = [k.strip() for k in m.group(1).split(',') if k.strip()]
        parts = []
        for k in keys:
            n = cite_num(k)
            parts.append('<a href="#ref-%d">%d</a>' % (n, n))
        return '<span class="cite">[%s]</span>' % ', '.join(parts)
    if allow_cite:
        t = re.sub(r'\\cite[tp]?\{([^}]*)\}', do_cite, t)
    else:
        t = re.sub(r'\\cite[tp]?\{([^}]*)\}', '', t)

    t = re.sub(r'\\(?:ref|autoref)\{([^}]*)\}', lambda m: '\x01' + m.group(1) + '\x01', t)
    t = re.sub(r'\\label\{[^}]*\}', '', t)

    # wrappers (innermost-out, repeat until stable)
    changed = True
    while changed:
        changed = False
        for name, (o, c) in WRAPPERS.items():
            m = re.search(r'\\%s\s*\{' % name, t)
            if m:
                body, end = find_group(t, t.index('{', m.start()))
                t = t[:m.start()] + o + body + c + t[end:]
                changed = True
    # {\bf ...} / {\it ...}
    t = re.sub(r'\{\\bf\s+([^{}]*)\}', r'<strong>\1</strong>', t)
    t = re.sub(r'\{\\it\s+([^{}]*)\}', r'<em>\1</em>', t)

    t = re.sub(r'\\url\{([^}]*)\}', r'<a href="\1">\1</a>', t)
    t = re.sub(r'\\href\{([^}]*)\}\{([^}]*)\}', r'<a href="\1">\2</a>', t)

    for pat, rep in SIMPLE:
        t = re.sub(pat, rep, t)

    # escapes and specials
    t = t.replace('\\%', '%').replace('\\&', '&amp;').replace('\\_', '_')
    t = t.replace('\\$', '$').replace('\\#', '#')
    t = t.replace('``', '&ldquo;').replace("''", '&rdquo;')
    t = re.sub(r'\\times\b', '&times;', t)
    t = accents(t)
    t = t.replace('~', '\u00a0')
    t = re.sub(r'\\\\', ' ', t)
    t = re.sub(r'\{\\L\}|\\L\b', '\u0141', t)
    t = re.sub(r'\{\\o\}|\\o\b', '\u00f8', t)
    t = t.replace('---', '&mdash;').replace('--', '&ndash;')
    t = re.sub(r'\\[a-zA-Z]+\s*', '', t)          # drop leftover control seqs
    t = t.replace('{', '').replace('}', '')
    t = re.sub(r'[ \t]+', ' ', t)
    t = restore_math(t, store)
    t = re.sub(r'\x01([^\x01]*)\x01', lambda m: ref_html(m.group(1)), t)
    return t.strip()

# ---------------------------------------------------------------- tabular
COLOR_CLASS = {'i1': 'best', 'i2': 'second', 'i3': 'third'}

def parse_tabular(body):
    # tabular, tabularx and tabular* all appear in this paper; the latter two take a
    # width argument before the column spec, which is dropped here.
    m = re.search(r'\\begin\{(tabularx|tabular\*|tabular)\}', body)
    if not m: return ''
    env = m.group(1)
    inner, _ = env_span(body, env, m.end())
    if env in ('tabularx', 'tabular*'):
        j = inner.index('{')
        _, k = find_group(inner, j)          # the width
        inner = inner[k:]
    # drop the column spec argument
    j = inner.index('{')
    spec, k = find_group(inner, j)
    inner = inner[k:]
    # \extracolsep and friends are layout glue, not columns
    spec = re.sub(r'@\{[^{}]*\}', '', spec)
    aligns = []
    for ch in re.sub(r'\{[^{}]*\}|[|@p*]', '', spec):
        aligns.append({'l': 'left', 'c': 'center', 'r': 'right',
                       'X': 'left', 'Y': 'left'}.get(ch, 'left'))

    rows = re.split(r'\\\\(?:\s*\[[^\]]*\])?', inner)
    out, header_done, rowsets = [], False, []
    for raw in rows:
        mrule = re.search(r'\\(midrule|Xhline|shline|specialrule)\b', raw)
        mtext = re.search(r'[A-Za-z0-9$\\]', re.sub(r'\\(toprule|midrule|bottomrule|hline|Xhline|shline|specialrule|cmidrule|rowcolor)\b(\{[^}]*\})*(\([^)]*\))?', '', raw))
        head_rule = bool(mrule)
        lead_rule = bool(mrule) and raw[:mrule.start()].strip() == ''
        raw = re.sub(r'\\(toprule|midrule|bottomrule|hline|shline)\b', '', raw)
        raw = re.sub(r'\\(Xhline|specialrule)\{[^}]*\}(\{[^}]*\})?', '', raw)
        raw = re.sub(r'\\cmidrule(\([^)]*\))?\{[^}]*\}', '', raw)
        if not raw.strip():
            if head_rule and out: header_done = True
            continue
        cells = re.split(r'(?<!\\)&', raw)
        tds = []
        ci = 0
        for c in cells:
            span, cls, al = 1, '', aligns[ci] if ci < len(aligns) else 'left'
            mg2 = re.search(r'\\grouprow\{(\d+)\}\s*\{', c)
            if mg2:
                inner_g, end_g = find_group(c, c.index('{', mg2.end() - 1))
                c = (c[:mg2.start()] + '\\multicolumn{%s}{l}{\\textit{%s}}'
                     % (mg2.group(1), inner_g) + c[end_g:])
            mm = re.search(r'\\multicolumn\{(\d+)\}\{([^}]*)\}\s*\{', c)
            if mm:
                span = int(mm.group(1))
                al = {'l': 'left', 'c': 'center', 'r': 'right'}.get(
                    re.sub(r'[|@]', '', mm.group(2))[:1], 'center')
                inner_c, end = find_group(c, c.index('{', mm.end() - 1))
                c = c[:mm.start()] + inner_c + c[end:]
            mc = re.search(r'\\cellcolor\{([^}]*)\}', c)
            if mc:
                cls = COLOR_CLASS.get(mc.group(1).split('!')[0], 'hl')
                c = c[:mc.start()] + c[mc.end():]
            for mg in re.finditer(r'\\(gain|drop|pos)\s*\{', c):
                pass
            c = re.sub(r'\\(gain|pos)\s*\{([^}]*)\}', r'<span class="gain">+\2</span>', c)
            c = re.sub(r'\\drop\s*\{([^}]*)\}', r'<span class="drop">\1</span>', c)
            tds.append((inline(c), span, cls, al))
            ci += span
        out.append((tds, head_rule, lead_rule))

    # first block until the first head_rule row is the header
    html_rows, in_head = [], True
    thead, tbody = [], []
    seen_rule = False
    for tds, head_rule, lead_rule in out:
        if lead_rule: seen_rule = True
        cell = 'th' if not seen_rule else 'td'
        r = '<tr>' + ''.join(
            '<%s%s%s class="%s">%s</%s>' % (
                cell, ' colspan="%d"' % s if s > 1 else '',
                '', (cls + ' ' + al).strip(), c, cell)
            for c, s, cls, al in tds) + '</tr>'
        (thead if not seen_rule else tbody).append(r)
        if head_rule: seen_rule = True
    if not tbody:
        tbody, thead = thead, []
    return ('<div class="tablewrap"><table>' +
            ('<thead>' + ''.join(thead) + '</thead>' if thead else '') +
            '<tbody>' + ''.join(tbody) + '</tbody></table></div>')

# ---------------------------------------------------------------- algorithm
def render_algorithm(body, num, label, caption):
    m = re.search(r'\\begin\{algorithmic\}', body)
    inner, _ = env_span(body, 'algorithmic', m.end())
    inner = re.sub(r'^\s*\[\d*\]', '', inner)
    lines, indent = [], 0
    ln = 0
    for raw in inner.split('\n'):
        raw = raw.strip()
        if not raw: continue
        dedent = re.match(r'\\(EndFor|EndIf|EndWhile|EndProcedure|Else)', raw)
        if dedent: indent = max(0, indent - 1)
        numbered = not raw.startswith('\\Statex')
        cmd = re.match(r'\\(State|Statex|For|EndFor|If|Else|EndIf|While|EndWhile|Return)\b', raw)
        txt = raw
        if cmd:
            kw = cmd.group(1)
            txt = raw[cmd.end():].strip()
            if kw == 'For': txt = '<b>for</b> ' + txt + ' <b>do</b>'
            elif kw == 'EndFor': txt = '<b>end for</b>'
            elif kw == 'If': txt = '<b>if</b> ' + txt + ' <b>then</b>'
            elif kw == 'EndIf': txt = '<b>end if</b>'
            elif kw == 'Else': txt = '<b>else</b>'
            elif kw == 'While': txt = '<b>while</b> ' + txt
            elif kw == 'EndWhile': txt = '<b>end while</b>'
        txt = re.sub(r'\\Comment\{([^}]*)\}', r'<span class="cmt">&#9655; \1</span>', txt)
        if numbered:
            ln += 1
            no = '<span class="ln">%d</span>' % ln
        else:
            no = '<span class="ln"></span>'
        lines.append('<div class="algline" style="padding-left:%dem">%s%s</div>'
                     % (indent * 2, no, inline(txt)))
        if re.match(r'\\(For|If|While|Procedure)\b', raw): indent += 1
    cap = '<div class="caption"><b>Algorithm %s.</b> %s</div>' % (num, inline(caption))
    return '<figure class="algo" id="%s">%s<div class="algobody">%s</div></figure>' % (
        label or 'alg%s' % num, cap, ''.join(lines))

# ---------------------------------------------------------------- figures
_amap = os.path.join(OUT, 'assets.json')
IMG = json.load(open(_amap)) if os.path.exists(_amap) else {}

def img_tag(path, cls=''):
    web = IMG.get(path)
    if web is None:
        return '<div class="missing">[missing figure: %s]</div>' % html.escape(path)
    return '<img class="%s" src="%s" alt="" loading="lazy">' % (cls, web)

def render_figure(body, num, label, caption):
    subs = []
    pos = 0
    while True:
        m = re.search(r'\\begin\{subfigure\}', body[pos:])
        if not m: break
        s = pos + m.end()
        inner, end = env_span(body, 'subfigure', s)
        pos = end
        # drop width args
        inner2 = inner
        mi = re.search(r'\\includegraphics(\[[^\]]*\])?\{([^}]*)\}', inner2)
        subcap = ''
        mc = re.search(r'\\caption\{', inner2)
        if mc:
            subcap, _ = find_group(inner2, inner2.index('{', mc.end() - 1))
        if mi:
            subs.append((mi.group(2), subcap))
    if subs:
        n = len(subs)
        cols = 3 if n % 3 == 0 else (2 if n % 2 == 0 else 3)
        items = ''.join(
            '<div class="subfig">%s<div class="subcap">%s</div></div>'
            % (img_tag(p), inline(c)) for p, c in subs)
        grid = '<div class="figgrid cols%d">%s</div>' % (cols, items)
    else:
        mi = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', body)
        grid = ''.join(img_tag(p, 'single') for p in mi)
    cap = '<figcaption><b>Fig. %s.</b> %s</figcaption>' % (num, inline(caption))
    return '<figure class="fig" id="%s">%s%s</figure>' % (label or 'fig%s' % num, grid, cap)

TABULAR_RE = re.compile(r'\\begin\{(tabularx|tabular\*|tabular)\}')

def table_panels(body):
    """Every tabular in the float, each with the \textbf{(a) ...} heading that
    precedes it.  Multi-panel tables are laid out side by side in the paper --
    in minipages, or stacked with a bold panel title -- and rendering only the
    first tabular silently drops the rest."""
    panels, pos = [], 0
    while True:
        m = TABULAR_RE.search(body, pos)
        if not m: break
        _, end = env_span(body, m.group(1), m.end())
        title, mt = '', None
        for cand in re.finditer(r'\\textbf\{', body[pos:m.start()]):
            mt = cand
        if mt:
            g, _ = find_group(body[pos:m.start()], mt.end() - 1)
            if g.lstrip().startswith('('):
                title = g
        panels.append((title, parse_tabular(body[m.start():end])))
        pos = end
    return panels

def render_table(body, num, label, caption):
    panels = table_panels(body)
    if not panels:
        panels = [('', parse_tabular(body))]
    tbl = ''.join(
        ('<div class="panelhead">%s</div>' % inline(t) if t else '') + h
        for t, h in panels)
    cap = '<div class="caption"><b>Table %s.</b> %s</div>' % (num, inline(caption))
    return '<figure class="tbl" id="%s">%s%s</figure>' % (label or 'tab%s' % num, cap, tbl)

# ---------------------------------------------------------------- float pass
FLOAT_ENVS = ['table*', 'figure*', 'table', 'figure', 'wraptable', 'wrapfigure', 'algorithm']

def extract_floats(t):
    floats = []
    out = []
    i = 0
    while i < len(t):
        m = re.compile(r'\\begin\{(table\*|figure\*|table|figure|wraptable|wrapfigure|algorithm)\}').search(t, i)
        if not m:
            out.append(t[i:]); break
        out.append(t[i:m.start()])
        name = m.group(1)
        body, end = env_span(t, name, m.end())
        floats.append((name, body))
        out.append('\n\n@@FLOAT%d@@\n\n' % (len(floats) - 1))
        i = end
    return ''.join(out), floats

def float_meta(body):
    caption = ''
    mcs = list(re.finditer(r'\\caption\{', body))
    mc = mcs[-1] if mcs else None
    if mc:
        caption, _ = find_group(body, mc.end() - 1)
    # the float's own label is the one that follows its (last) \caption
    tail = body[mc.end():] if mc else body
    ml = re.search(r'\\label\{([^}]*)\}', tail) or re.search(r'\\label\{([^}]*)\}', body)
    label = ml.group(1) if ml else ''
    return caption, label

# ---------------------------------------------------------------- body pass
def render_body(t, floats):
    """t has section commands, paragraphs and @@FLOATn@@ markers."""
    # protect display equations
    eqs = []
    def grab_eq(m):
        name = m.group(1)
        inner, end = env_span(t_holder[0], name, m.end())
        return inner
    # simpler: regex over equation environments (they don't nest here)
    def eq_rep(m):
        eqs.append(m.group(0))
        return '\n\n@@EQ%d@@\n\n' % (len(eqs) - 1)
    t = re.sub(r'\\begin\{(equation|align|gather)\*?\}.*?\\end\{\1\*?\}', eq_rep, t, flags=re.S)

    blocks = []
    # split on sectioning commands
    parts = re.split(r'(\\(?:sub)*section\*?\s*\{)', t)
    # rebuild: parts[0] is preamble text, then pairs
    stream = [('text', parts[0])]
    i = 1
    while i < len(parts):
        cmd = parts[i]
        rest = parts[i+1]
        depth = cmd.count('sub')
        title, k = find_group('{' + rest, 0)
        after = rest[k-1:]
        stream.append(('sec%d%s' % (depth, '*' if '*' in cmd else ''), title))
        stream.append(('text', after))
        i += 2

    html_out = []
    toc = []
    counters = {'sec': 0, 'sub': 0, 'subsub': 0}
    fignum = [0]; tabnum = [0]; algnum = [0]
    cur_sec = ['S0', 'top']

    def flush_text(txt):
        # paragraphs
        for para in re.split(r'\n\s*\n', txt):
            para = para.strip()
            if not para: continue
            mf = re.fullmatch(r'@@FLOAT(\d+)@@', para)
            if mf:
                html_out.append(render_float(int(mf.group(1))))
                continue
            me = re.fullmatch(r'@@EQ(\d+)@@', para)
            if me:
                body = eqs[int(me.group(1))]
                html_out.append('<div class="eqn">%s</div>' % html_escape_math(body))
                continue
            ml = re.match(r'\\label\{([^}]*)\}', para)
            if ml:
                LABELS[ml.group(1)] = ('sec', cur_sec[0], cur_sec[1])
                para = para[ml.end():].strip()
                if not para:
                    continue
            # boldparagraph
            mb = re.match(r'\\boldparagraph\s*\{', para)
            lead = ''
            if mb:
                title, k = find_group(para, para.index('{', mb.end() - 1))
                title = inline(title).rstrip('.').rstrip()
                lead = '<strong class="bp">%s.</strong> ' % title
                para = para[k:]
            html_out.append('<p>%s%s</p>' % (lead, inline(para)))

    def render_float(idx):
        name, body = floats[idx]
        caption, label = float_meta(body)
        if name.startswith('figure') or name.startswith('wrapfigure'):
            fignum[0] += 1
            num = 'S%d' % fignum[0]
            if label: LABELS[label] = ('fig', num, label)
            return render_figure(body, num, label, caption)
        if name == 'algorithm':
            algnum[0] += 1
            num = 'S%d' % algnum[0]
            if label: LABELS[label] = ('alg', num, label)
            return render_algorithm(body, num, label, caption)
        tabnum[0] += 1
        num = 'S%d' % tabnum[0]
        if label: LABELS[label] = ('tab', num, label)
        return render_table(body, num, label, caption)

    for kind, val in stream:
        if kind == 'text':
            flush_text(val)
        else:
            starred = kind.endswith('*')
            depth = int(kind.rstrip('*')[-1])
            if starred:
                num = ''
                anchor = 'sec-' + re.sub(r'[^a-z0-9]+', '-', inline(val).lower()).strip('-')
                html_out.append('@@SEC:%d:%s:%s:%s@@' % (depth, num, anchor, inline(val)))
                toc.append((depth, num, anchor, inline(val)))
                cur_sec[0], cur_sec[1] = num, anchor
                continue
            if depth == 0:
                counters['sec'] += 1; counters['sub'] = 0; counters['subsub'] = 0
                num = 'S%d' % counters['sec']
            elif depth == 1:
                counters['sub'] += 1; counters['subsub'] = 0
                num = 'S%d.%d' % (counters['sec'], counters['sub'])
            else:
                counters['subsub'] += 1
                num = 'S%d.%d.%d' % (counters['sec'], counters['sub'], counters['subsub'])
            anchor = 'sec-' + num.lower().replace('.', '-')
            html_out.append('@@SEC:%d:%s:%s:%s@@' % (depth, num, anchor, inline(val)))
            toc.append((depth, num, anchor, inline(val)))
            cur_sec[0], cur_sec[1] = num, anchor
    return html_out, toc

# ---------------------------------------------------------------- driver
def expand_inputs(path, depth=0):
    """Inline \\input{...} so the multi-file supplement reads as one source."""
    assert depth < 8, 'input nesting too deep at ' + path
    txt = open(path).read()
    def sub(m):
        child = m.group(1)
        # style/ and tables/numbers are the preamble and the prose-number macros;
        # inlining them dumps the whole package list into the page body.
        if child.startswith('style/') or child in ('tables/numbers', 'tables/numbers.tex'):
            return ''
        if not child.endswith('.tex'):
            child += '.tex'
        f = os.path.join(PAPER, child)
        return expand_inputs(f, depth + 1) if os.path.exists(f) else ''
    return re.sub(r'\\input\{([^}]*)\}', sub, txt)

NUMBERS = {}

def load_numbers(path):
    """tables/numbers.tex is the single source for every number quoted in prose.
    It is not inlined into the page (it is macro definitions, not text), so the
    macros are expanded here instead -- otherwise \\numRoutes{} silently
    disappears from a caption and the sentence reads as if the count were never
    written."""
    if not os.path.exists(path): return
    txt = open(path).read()
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\{', txt):
        body, _ = find_group(txt, m.end() - 1)
        NUMBERS[m.group(1)] = body

def expand_numbers(t):
    for _ in range(4):                      # macros may cite other macros
        new = re.sub(r'\\([A-Za-z]+)\{\}|\\([A-Za-z]+)(?![A-Za-z])',
                     lambda m: NUMBERS.get(m.group(1) or m.group(2), m.group(0)), t)
        if new == t: break
        t = new
    return t

def main():
    load_numbers(os.path.join(PAPER, 'tables', 'numbers.tex'))
    src = expand_numbers(expand_inputs(os.path.join(PAPER, 'supplementary.tex')))
    src = strip_comments(src)
    if '\\begin{document}' in src:
        src = src.split('\\begin{document}', 1)[1].split('\\end{document}')[0]
    for cmd in ('maketitle', 'raggedbottom', 'clearpage', 'newpage'):
        src = re.sub(r'\\%s\b' % cmd, '', src)
    src = re.sub(r'\\(this)?pagestyle\{[^}]*\}', '', src)
    src = re.sub(r'\\(title|author)\{', r'\\hidden{', src)
    src = re.sub(r'\\bibliography(style)?\{[^}]*\}', '', src)
    # remove the LaTeX table of contents scaffolding
    src = re.sub(r'\\begingroup.*?\\endgroup', '', src, flags=re.S)
    src = re.sub(r'\\addtocontents\{[^}]*\}\{[^\n]*\}', '', src)
    src = re.sub(r'\\setcounter\{[^}]*\}\{[^}]*\}', '', src)
    src = re.sub(r'\\tableofcontents', '', src)
    src = re.sub(r'\\hypersetup\{[^}]*\}', '', src)

    body, floats = extract_floats(src)
    # two passes so that \ref{} to floats resolves (labels filled on pass 1)
    for _ in range(2):
        CITE_ORDER.clear()
        html_out, toc = render_body(body, floats)
    json.dump({k: v for k, v in LABELS.items()}, open(os.path.join(OUT, 'labels.json'), 'w'), indent=1)

    # substitute section markers
    def sec_marker(m):
        depth, num, anchor, title = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        tag = 'h2' if depth == 0 else ('h3' if depth == 1 else 'h4')
        numhtml = '<span class="num">%s</span>' % num if num else ''
        return ('<%s id="%s" class="secnum">%s%s</%s>'
                % (tag, anchor, numhtml, title, tag))
    out = '\n'.join(html_out)
    out = re.sub(r'@@SEC:(\d+):([^:]*):([^:]*):(.*?)@@', sec_marker, out, flags=re.S)

    # references
    refs = []
    for i, key in enumerate(CITE_ORDER, 1):
        txt = BBL.get(key, '<em>missing bib entry: %s</em>' % key)
        refs.append('<li id="ref-%d"><span class="rn">[%d]</span> %s</li>' % (i, i, inline(txt, allow_cite=False)))
    refs_html = '<ol class="refs">%s</ol>' % ''.join(refs)

    toc_html = []
    for depth, num, anchor, title in toc:
        toc_html.append('<li class="d%d"><a href="#%s"><span class="num">%s</span>%s</a></li>'
                        % (depth, anchor, num, title))
    toc_html = '<ul class="toc">%s</ul>' % ''.join(toc_html)

    open(os.path.join(OUT, 'body.html'), 'w').write(out)
    open(os.path.join(OUT, 'toc.html'), 'w').write(toc_html)
    open(os.path.join(OUT, 'refs.html'), 'w').write(refs_html)
    missing = re.findall(r'\[missing figure: ([^\]]*)\]', out)
    print('sections:', len(toc), 'citations:', len(CITE_ORDER))
    print('missing figures:', sorted(set(missing)))
    print('unresolved refs:', len(re.findall(r'>\?<', out)))

if __name__ == '__main__':
    main()
