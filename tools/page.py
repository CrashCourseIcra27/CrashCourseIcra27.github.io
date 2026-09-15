#!/usr/bin/env python3
"""Assemble site/index.html from the converted body, TOC and references."""
import os, json, shutil, subprocess

OUT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.environ.get('CC_PAPER_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
SITE = os.environ.get('CC_SITE_DIR', os.path.dirname(OUT))

body = open(os.path.join(OUT, 'body.html')).read()
toc = open(os.path.join(OUT, 'toc.html')).read()
refs = open(os.path.join(OUT, 'refs.html')).read()
macros = json.load(open(os.path.join(OUT, 'macros.json')))

TITLE = 'CrashCourse: Learning Emergency Maneuvers from Privileged Driving Teachers'



BIBTEX = """@inproceedings{crashcourse2027,
  title     = {CrashCourse: Learning Emergency Maneuvers from Privileged Driving Teachers},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}"""

CSS = r"""
:root{
  --bg:#ffffff; --fg:#1a1c1f; --muted:#5c6470; --line:#e3e6ea; --soft:#f6f7f9;
  --accent:#1f5fd0; --accent-soft:#e8f0fd; --best:#dff3e3; --second:#eef3fb;
  --gain:#dff3e3; --drop:#fbe4e4; --maxw:860px;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#14161a; --fg:#e7e9ec; --muted:#9aa3ae; --line:#2b3038; --soft:#1b1e24;
         --accent:#7ba7f0; --accent-soft:#1e2836; --best:#1e3326; --second:#1d2735;
         --gain:#1e3326; --drop:#3a2323; }
}
*{box-sizing:border-box}
html{scroll-behavior:smooth; scroll-padding-top:1.2rem}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
     -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 20px}

/* ---------- hero ---------- */
header.hero{padding:56px 0 28px;border-bottom:1px solid var(--line);background:var(--soft)}
.kicker{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;color:var(--muted);
        font-weight:600;margin-bottom:14px}
h1.title{font-size:2.05rem;line-height:1.22;margin:0 0 18px;font-weight:700;letter-spacing:-.015em}
.authors{color:var(--muted);font-size:.95rem;margin-bottom:6px}
.venue{color:var(--muted);font-size:.9rem;margin-bottom:22px}
.buttons{display:flex;flex-wrap:wrap;gap:10px}
.btn{display:inline-flex;align-items:center;gap:7px;padding:8px 16px;border-radius:999px;
     background:var(--fg);color:var(--bg);font-size:.88rem;font-weight:500}
.btn:hover{opacity:.85;text-decoration:none}
.btn.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
.btn.dim{opacity:.45;pointer-events:none}

/* ---------- generic sections ---------- */
section{padding:34px 0}
section+section{border-top:1px solid var(--line)}
h2,h3,h4{line-height:1.3;font-weight:650;letter-spacing:-.01em}
h2{font-size:1.4rem;margin:6px 0 14px}
h3{font-size:1.12rem;margin:30px 0 10px}
h4{font-size:1rem;margin:22px 0 8px}
h2 .num,h3 .num,h4 .num{color:var(--muted);font-weight:500;margin-right:.6em;font-variant-numeric:tabular-nums}
p{margin:0 0 14px}
.lead{font-size:1.02rem}
strong.bp{font-weight:650}
.sc{font-variant:small-caps}
.note{color:#b04a4a}
.status{border:1px solid #e2c391;background:#fdf6e8;border-radius:10px;padding:14px 16px;
        font-size:.94rem}
@media (prefers-color-scheme: dark){.status{background:#2a2317;border-color:#5c4a2a}}
.pend{color:#b04a4a;font-weight:700}
table td .pend,table th .pend{font-weight:700}

.teaser{margin:26px 0 0}
.teaser img{width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
.teaser figcaption{color:var(--muted);font-size:.85rem;margin-top:10px}

/* ---------- toc ---------- */
ul.toc{list-style:none;margin:0;padding:0;columns:2;column-gap:34px}
@media (max-width:640px){ul.toc{columns:1}}
ul.toc li{break-inside:avoid;margin:0 0 5px;font-size:.92rem}
ul.toc li.d0{margin-top:10px;font-weight:600}
ul.toc li.d1{padding-left:16px}
ul.toc li.d2{padding-left:32px;font-size:.88rem}
ul.toc .num{color:var(--muted);display:inline-block;min-width:3.1em;font-variant-numeric:tabular-nums}
ul.toc a{color:var(--fg)}

/* ---------- floats ---------- */
figure{margin:26px 0}
figure .caption,figcaption{color:var(--muted);font-size:.86rem;line-height:1.55;margin-top:10px}
figure.tbl .caption{margin:0 0 10px}
figure img{max-width:100%;height:auto;display:block;border-radius:6px}
figure img.single{width:100%;border:1px solid var(--line);background:#fff;padding:6px}
.figgrid{display:grid;gap:8px}
.figgrid.cols3{grid-template-columns:repeat(3,1fr)}
.figgrid.cols2{grid-template-columns:repeat(2,1fr)}
@media (max-width:600px){.figgrid.cols3,.figgrid.cols2{grid-template-columns:1fr 1fr}}
.subfig img{width:100%;border:1px solid var(--line)}
.subcap{color:var(--muted);font-size:.78rem;text-align:center;margin-top:4px}
.missing{padding:10px;border:1px dashed var(--line);color:var(--muted);font-size:.85rem}

pre.tree{background:var(--soft);border:1px solid var(--line);border-radius:8px;
          padding:12px 14px;overflow-x:auto;font-size:.82rem;line-height:1.5;margin:14px 0}

/* ---------- tables ---------- */
.panelhead{font-weight:650;font-size:.9rem;margin:14px 0 6px}
.panelhead:first-of-type{margin-top:0}
.tbl .tablewrap + .panelhead{margin-top:18px}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);
           border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.87rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{background:var(--soft);font-weight:650;border-bottom:2px solid var(--line)}
tbody tr:last-child td{border-bottom:none}
td.left,th.left{text-align:left}
td.center,th.center{text-align:center}
td.right,th.right{text-align:right}
td.best{background:var(--best);font-weight:650}
td.second{background:var(--second)}
td.hl{background:var(--accent-soft)}
.gain{background:var(--gain);border-radius:4px;padding:0 4px;font-size:.8em}
.drop{background:var(--drop);border-radius:4px;padding:0 4px;font-size:.8em}
.yes{color:#2a8a4a;font-weight:700}
.no{color:#c05252;font-weight:700}

/* ---------- equations / algorithms ---------- */
.eqn{overflow-x:auto;overflow-y:hidden;margin:16px 0;padding:2px 0}
figure.algo{border:1px solid var(--line);border-radius:8px;padding:14px 16px;background:var(--soft)}
figure.algo .caption{margin:0 0 10px;color:var(--fg);font-size:.9rem;
                     border-bottom:1px solid var(--line);padding-bottom:8px}
.algobody{font-size:.85rem;overflow-x:auto}
.algline{padding-top:2px;white-space:nowrap}
.algline .ln{display:inline-block;width:2.1em;color:var(--muted);
             font-variant-numeric:tabular-nums;font-size:.8em}
.cmt{color:var(--muted)}

/* ---------- refs / cites ---------- */
.cite a{font-size:.86em}
.mainref{font-weight:600}
ol.refs{list-style:none;margin:0;padding:0;font-size:.85rem;color:var(--muted)}
ol.refs li{margin:0 0 7px;padding-left:2.6em;text-indent:-2.6em;line-height:1.5}
ol.refs .rn{display:inline-block;width:2.4em;text-indent:0;color:var(--fg)}
ol.refs li:target{background:var(--accent-soft);border-radius:4px}

pre.bibtex{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px;
           overflow-x:auto;font-size:.82rem;line-height:1.5}
footer{padding:34px 0 60px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line)}
mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden}
"""

MATHJAX = """
window.MathJax = {
  loader: { load: ['[tex]/tagformat'] },
  tex: {
    packages: { '[+]': ['tagformat'] },
    inlineMath: [['$','$'],['\\\\(','\\\\)']],
    displayMath: [['$$','$$'],['\\\\[','\\\\]']],
    processEnvironments: true,
    tags: 'ams',
    tagformat: { number: (n) => 'S' + n },
    macros: Object.assign(%s, {num:['{#1}',1], si:['{#1}',1], SI:['{#1}\\\\,{#2}',2]})
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] },
  svg: { fontCache: 'global' }
};
""" % json.dumps(macros)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="Project page for CrashCourse: Learning Emergency Maneuvers from Privileged Driving Teachers.">
<meta property="og:title" content="{title}">
<meta property="og:description" content="Project page: benchmark scenarios, implementation details, and the evidence registry.">
<meta property="og:image" content="static/images/figures_teaser.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#128680;</text></svg>">
<style>{css}</style>
<script>{mathjax}</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <div class="kicker">Project Page</div>
    <h1 class="title">{title}</h1>
    <div class="authors">Anonymous Authors</div>
    <div class="venue">Under double-anonymous review at ICRA 2027</div>
    <div class="buttons">
      <a class="btn ghost dim" href="#">&#128196; Paper (coming soon)</a>
      <a class="btn" href="crashcourse-corpus.zip">&#128230; Benchmark corpus (1.4&#8239;MB)</a>
      <a class="btn ghost dim" href="#">&#127909; Video (coming soon)</a>
    </div>
    <figure class="teaser">
      <img src="static/images/figures_teaser.png" alt="One CrashCourse encounter driven by three policies">
      <figcaption>One documented CrashCourse encounter, driven from the same onset by the
      released sensor policy, by the privileged teacher, and by the same student after
      adapter-only fine-tuning on the teacher's demonstrations. The hatched panels are the
      two rollouts that have not been run yet.</figcaption>
    </figure>
  </div>
</header>

<section id="status">
  <div class="wrap">
    <div class="status">
      <b>Work in progress.</b> The teacher results are measured: MU-SAC reaches
      <b>53.11 DS</b> at <b>1.19 collisions/km</b> on CrashCourse, against 35.64 DS for the
      strongest released camera policy and 39.35 for privileged PDM-Lite. The student
      transfer experiments are still running, and every cell still marked
      <span class="pend">P</span> here and in the paper is a pending run, not a measured
      zero.
    </div>
  </div>
</section>

<section id="abstract">
  <div class="wrap">
    <h2>Abstract</h2>
    <p class="lead">Emergency-driving competence depends both on learning effective responses
    and on obtaining demonstrations that a sensor-based policy can use. We study these two
    questions in CrashCourse, a crash-grounded closed-loop simulation suite. First, we train a
    privileged driving teacher with MU-SAC, using distributional critics, multiplicative
    steering conditioning and uncertainty-weighted replay; the teacher learns by reinforcement
    without a behavior-cloning initialization or reference policy. Second, we construct a small
    dataset of synchronized sensor observations and emergency-response targets from its
    training-side rollouts, and adapt off-the-shelf driving models by supervised fine-tuning of
    added adapters while their original weights stay frozen. Matched teacher ablations test the
    RL method, while nominal-only adaptation, data-source controls and encounter-budget sweeps
    test the value of the CrashCourse demonstrations. Ordinary-driving evaluation measures
    regression rather than assuming that frozen base weights prevent it.</p>
  </div>
</section>

<section id="method">
  <div class="wrap">
    <h2>Teacher to student</h2>
    <p>MU-SAC trains the privileged teacher in the reactive CrashCourse environment. A frozen
    checkpoint is rolled out on the training split only, and each accepted encounter is
    exported with synchronized student sensors, causal history and native targets. Released
    sensor policies are then adapted by supervised fine-tuning of added adapters, with their
    original weights frozen. Privileged simulator state never crosses into the deployed
    student: what transfers is observations and teacher actions.</p>
    <figure class="teaser">
      <img src="static/images/figures_pipeline.png" alt="The three-stage teacher-to-student pipeline">
    </figure>
  </div>
</section>

<section id="benchmark">
  <div class="wrap">
    <h2>The benchmark</h2>
    <p>CrashCourse scenarios are built from real crash geometries and replayed in closed loop
    with reactive adversaries: an adversary tracks the ego vehicle every frame to preserve the
    intended conflict geometry, follows explicit velocity control once the ego crosses the
    trigger, and returns to simulator autopilot afterwards, so the conflict is reproducible
    without the aftermath being scripted. The scenario families, towns and conditions are
    listed in <a href="#content">the appendix material</a> below.</p>
    <figure class="teaser">
      <img src="static/images/figures_benchmark_design.png" alt="CrashCourse scenario construction">
    </figure>
    <p>Two measurements separate it from the closest broad benchmark: 33% of CrashCourse's
    actor interactions leave the ego under two seconds to react, against 14% of
    Bench2Drive's, and the impacts move to the front of the vehicle — 65% frontal against
    13%. Bench2Drive penalizes a policy for being struck while it hesitates; CrashCourse
    penalizes it for driving into a conflict it cannot resolve.</p>
    <figure class="teaser">
      <img src="static/images/figures_benchmark_stats.png" alt="Time-to-collision and impact-bearing against Bench2Drive">
      <figcaption>(a) NPC-to-ego time-to-collision and NPC speed distributions.
      (b) Ego collision-bearing density; both panels share one normalization constant.</figcaption>
    </figure>
  </div>
</section>

<section id="release">
  <div class="wrap">
    <h2>The benchmark corpus</h2>
    <p>All 164 evaluated scenario instances are released here:
    <a href="crashcourse-corpus.zip">crashcourse-corpus.zip</a> (1.4&#8239;MB), or browse an
    instance directly, for example
    <a href="corpus/manifest.csv">manifest.csv</a> and
    <a href="corpus/README.md">README.md</a>.</p>

    <p>Each instance is one folder holding its <code>route.xml</code>, its scenario
    implementation, and the extra behaviours it needs. Folders are self-contained and
    duplicate shared code on purpose, so no instance depends on another and any subset can be
    installed on its own. Every scenario class and module name is unique across the corpus,
    which is what lets a family sited in several towns keep a per-town implementation without
    one route silently loading a sibling town&rsquo;s version.</p>

    <pre class="tree">train/&lt;scenario&gt;__&lt;Town&gt;[__variantN]__wx&lt;1|2&gt;_&lt;WeatherPreset&gt;/
    route.xml                     route definition
    &lt;scenario&gt;_&lt;town&gt;.py           the scenario implementation
    custom_atomics_&lt;stem&gt;.py      extra behaviours, where required
test/ ...
manifest.csv                      every instance: split, town, preset, class, module</pre>

    <p>The split is per route, so both weather variants of a route land on the same side:
    82 train instances and 82 test instances. Weather is constant along a route, written into
    <code>route.xml</code> as the exact nine parameters of the named
    <code>carla.WeatherParameters</code> preset. In CARLA 0.9.15 route mode
    <code>precipitation_deposits</code> and <code>wetness</code> are visual only and do not
    change tyre grip; the instances that genuinely reduce grip do so in code, through a
    friction trigger. Some scenarios govern the ego&rsquo;s speed so the hazard is met at a
    repeatable closing speed, which is a property of the harness rather than of the hazard.</p>

    <p>Running an instance needs CARLA 0.9.15 with ScenarioRunner and the leaderboard: copy
    the instance&rsquo;s scenario module into <code>srunner/scenarios/</code>, copy any
    <code>custom_atomics_*.py</code> into
    <code>srunner/scenariomanager/scenarioatomics/</code>, and point the leaderboard at the
    instance&rsquo;s <code>route.xml</code>. Beyond a standard install the modules import only
    <code>carla</code>, <code>py_trees</code>, <code>numpy</code>, <code>cv2</code> and stock
    <code>srunner</code> helpers.</p>

    <p>The teacher training code, the exported demonstrations and the qualitative video follow
    once the runs they document are complete. Nothing is withheld pending review beyond the
    paper PDF itself, which goes up when review concludes.</p>
  </div>
</section>

<section id="contents">
  <div class="wrap">
    <h2>Contents</h2>
    {toc}
  </div>
</section>

<section id="content">
  <div class="wrap">
{body}
  </div>
</section>

<section id="references">
  <div class="wrap">
    <h2>References</h2>
    {refs}
  </div>
</section>

<section id="bibtex">
  <div class="wrap">
    <h2>BibTeX</h2>
    <pre class="bibtex">{bibtex}</pre>
  </div>
</section>

<footer>
  <div class="wrap">
    Numbers prefixed with S refer to this page; plain numbers refer to the main paper.
    This page is the project's extra material. ICRA has no supplementary-material track, so
    nothing here is required to follow the 8-page paper.
  </div>
</footer>

</body>
</html>
"""

def main():
    os.makedirs(SITE, exist_ok=True)
    html = HTML.format(title=TITLE, css=CSS, mathjax=MATHJAX,
                       toc=toc, body=body, refs=refs, bibtex=BIBTEX)
    open(os.path.join(SITE, 'index.html'), 'w').write(html)
    open(os.path.join(SITE, '.nojekyll'), 'w').write('')
    os.makedirs(os.path.join(SITE, 'static', 'videos'), exist_ok=True)
    open(os.path.join(SITE, 'static', 'videos', '.gitkeep'), 'w').write('')
    print('wrote', os.path.join(SITE, 'index.html'),
          os.path.getsize(os.path.join(SITE, 'index.html')), 'bytes')

if __name__ == '__main__':
    main()
