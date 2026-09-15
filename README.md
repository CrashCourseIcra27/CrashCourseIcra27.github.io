# CrashCourseIcra27.github.io

Project page for **CrashCourse: Learning Emergency Maneuvers from Privileged Driving
Teachers**, under double-anonymous review at ICRA 2027. It hosts the benchmark scenario
descriptions, implementation and reproducibility details, and the evidence registry —
the material the 8-page paper points at but does not contain. The paper links here from
the end of its abstract, so this page is the only published home for that material.

**ICRA has no supplementary material.** Everything that carries the argument lives in the
8-page PDF; this page is an optional extra that reviewers are not obliged to open, and it
must not be load-bearing. It is framed as a *project page*, never as a supplement.

**The study is unfinished, and the page says so.** A status banner sits above the abstract,
and every unmeasured cell renders as a red `P`, matching the paper. Do not quietly drop the
banner or blank those cells: an empty table reads as a measured zero.

## Layout

```
index.html               the whole page (self-contained CSS, MathJax from CDN)
static/images/           every figure, rasterized for the web
tools/                   the LaTeX -> HTML build scripts
.nojekyll                tells GitHub Pages to serve the files as-is
```

## Publishing

Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
The site then serves at `https://crashcourseicra27.github.io/`.

The repo name must stay `CrashCourseIcra27.github.io`, matching the org. GitHub serves an org
page only when the repo is named `<org>.github.io`; under any other name it becomes a
*project* page at `https://crashcourseicra27.github.io/<repo>/`, and the URL printed in the
paper 404s.

**Note:** GitHub Pages publishes from a *private* repository only on a paid plan
(Pro / Team / Enterprise). On a free plan the repo has to be public for the URL to go live.
The files can sit here privately until you are ready either way.

## Rebuilding from the LaTeX source

The page is generated from `supplementary.tex` and the `supp/*.tex` it inputs, in the paper
directory one level up. From this directory:

```bash
python3 tools/assets.py    # rasterize every referenced figure into static/images/
python3 tools/build.py     # supplementary.tex -> tools/{body,toc,refs}.html
python3 tools/page.py      # stitch everything into index.html
```

`tools/build.py` resolves cross-references, citations (from `supplementary.bbl`), tables,
equations and floats. References into the main paper are resolved from
`main.aux`, so build the paper first:

```bash
cd .. && make main
```

A stale or missing `.aux` leaves those numbers unresolved; `tools/build.py` reports the
count at the end of its run, and it should print `unresolved refs: 0`.

The scripts read the paper from `$CC_PAPER_DIR`, defaulting to the parent directory. Point
it wherever your checkout lives:

```bash
CC_PAPER_DIR=/path/to/story2_teacher_transfer_paper python3 tools/build.py
```

Never hard-code that path back into the scripts: this repo is the public artifact, and an
absolute path is an identifier.

## Anonymity

Check every drop before committing — this repo is the public artifact and the paper is under
double-anonymous review:

```bash
grep -rIl "$(whoami)" .                                     # your username
grep -rIhoE '/[a-z][a-z0-9_]*/[A-Za-z0-9_.-]+' . | sort -u  # absolute paths
grep -rIhoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' . | sort -u
```

Anything in those lists that is not upstream third-party code has to go: username, surname,
institution, cluster mount points, experiment-tracker entity. The same check applies to any
code or video archive added later, unzipped first.

## Paper PDF, code and video

The paper PDF is not published while the paper is under review. To add it once review
concludes, copy it into the repo root as `main.pdf` and point the disabled
"Paper (coming soon)" button in `index.html` at it. The same holds for the code archive and
the qualitative video: the buttons are in place and disabled, and the `#release` section
says what is coming.
