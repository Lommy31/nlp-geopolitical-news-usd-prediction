"""Generate figures/drawio/pipeline.drawio (editable) in a draw.io paper style.
Run from repo root: python3 report/eda-visualization/make_pipeline_drawio.py
Then export: drawio -x -f pdf --crop -o report/figures/assets/pipeline.pdf report/figures/drawio/pipeline.drawio"""
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[2]
cells, n = [], [1]
def nid():
    n[0] += 1; return f"c{n[0]}"

BASE = "html=1;whiteSpace=wrap;strokeColor=#000000;strokeWidth=1;fillColor=#FFFFFF;fontColor=#000000;fontFamily=Helvetica;"
def v(x, y, w, h, text="", style="", fs=10, rounded=True, italic=False, bold=False, dashed=False, fill=None, stroke=None, id=None):
    i = id or nid()
    s = BASE + f"fontSize={fs};"
    if rounded: s += "rounded=1;arcSize=12;"
    fs_style = (2 if italic else 0) + (1 if bold else 0)
    if fs_style: s += f"fontStyle={fs_style};"
    if dashed: s += "dashed=1;"
    if fill: s += f"fillColor={fill};"
    if stroke: s += f"strokeColor={stroke};"
    s += style
    cells.append(f'<mxCell id="{i}" value={quoteattr(text)} style="{s}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def text(x, y, w, h, t, color="#000000", fs=10, italic=False, bold=False, align="center"):
    i = nid()
    st = (2 if italic else 0) + (1 if bold else 0)
    s = f"text;html=1;whiteSpace=wrap;fontSize={fs};fontFamily=Helvetica;fontColor={color};align={align};verticalAlign=middle;strokeColor=none;fillColor=none;" + (f"fontStyle={st};" if st else "")
    cells.append(f'<mxCell id="{i}" value={quoteattr(t)} style="{s}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def trap(x, y, w, h, t, fs=9, fill="#FFFFFF"):
    i = nid()
    s = BASE + f"shape=trapezoid;perimeter=trapezoidPerimeter;direction=south;fixedSize=1;size=22;fontSize={fs};fontStyle=2;fillColor={fill};"
    cells.append(f'<mxCell id="{i}" value={quoteattr(t)} style="{s}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def e(a, b, dashed=False, color="#000000", label="", ex=None, ey=None, nx=None, ny=None, pts=None):
    i = nid()
    s = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=1;fontSize=9;fontFamily=Helvetica;" + f"strokeColor={color};"
    if dashed: s += "dashed=1;"
    for k, val in (("exitX", ex), ("exitY", ey), ("entryX", nx), ("entryY", ny)):
        if val is not None: s += f"{k}={val};"
    g = '<mxGeometry relative="1" as="geometry"/>' if not pts else '<mxGeometry relative="1" as="geometry"><Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + "</Array></mxGeometry>"
    cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style="{s}" edge="1" parent="1" source="{a}" target="{b}">{g}</mxCell>')

def lane(x, y, w, h):
    v(x, y, w, h, "", "arcSize=6;", dashed=True)

# ---------------- inputs
mk = v(10, 55, 125, 62, "Market features\n(14 inputs, data up to day t)", fs=10)
nw = v(10, 232, 125, 220, "News articles\n(headline + first 180 words)", fs=10)

# ---------------- Level 1
lane(150, 10, 715, 145)
text(160, 14, 200, 18, "Level 1  Baselines", fs=10, bold=True, align="left")
x1 = v(172, 48, 62, 26, "x")
a1 = v(172, 82, 62, 26, "a (optional)", dashed=True, fs=9)
t1 = v(172, 116, 62, 26, "t (optional)", dashed=True, fs=9)
cc1 = v(262, 48, 44, 94, "Concat", fs=9)
d1 = v(332, 76, 84, 40, "Dense 16\nReLU", italic=True)
p1 = v(442, 80, 56, 32, "P(UP)")
for a, b in [(x1, cc1), (a1, cc1), (t1, cc1), (cc1, d1), (d1, p1)]: e(a, b)
v(540, 34, 310, 104, "", "arcSize=10;", fill="#B0E3E6", stroke="#0E8088")
text(545, 36, 300, 16, "No learned text model", color="#0E8088", fs=9, italic=True)
v(556, 66, 130, 40, "Training majority", italic=True)
v(702, 66, 130, 40, "Last observed direction", italic=True)
text(556, 112, 280, 18, "t: mean FinBERT vector of the day", fs=8, italic=True)
e(mk, x1)
_p1=p1

# ---------------- Level 2
lane(150, 168, 715, 168)
text(160, 172, 360, 18, "Level 2  Text representation (one encoder at a time)", fs=10, bold=True, align="left")
ar2 = [v(172, 200 + k * 36, 62, 28, t, fs=9, italic=True) for k, t in enumerate(["Article 1", "Article 2", "Article n"])]
enc2 = trap(258, 198, 110, 112, "Frozen encoder\nTF-IDF + SVD,\nWord2Vec, FinBERT,\nRoBERTa-base", fs=8)
es2 = [v(392, 200 + k * 36, 34, 28, t, fs=9) for k, t in enumerate(["e1", "e2", "en"])]
mean2 = v(448, 214, 56, 58, "Mean", italic=True)
t2 = v(528, 228, 38, 28, "t")
xa2 = v(528, 280, 38, 28, "x, a", fs=9)
cc2 = v(596, 206, 44, 104, "Concat", fs=9)
d2 = v(668, 238, 84, 40, "Dense 16\nReLU", italic=True)
p2 = v(778, 242, 56, 32, "P(UP)")
for a in ar2: e(a, enc2)
for b in es2: e(enc2, b)
for b in es2: e(b, mean2)
e(mean2, t2); e(t2, cc2, ey=.5, nx=0, ny=.3); e(xa2, cc2, nx=0, ny=.85); e(cc2, d2); e(d2, p2)
text(676, 176, 180, 16, "Concatenation", color="#2E6FB5", fs=9, italic=True, align="right")
text(540, 312, 300, 16, "text-only variant: x zeroed, a kept", fs=8, italic=True)

# ---------------- Level 3
lane(150, 350, 715, 166)
text(160, 354, 360, 18, "Level 3  Fusion variants (FinBERT fixed)", fs=10, bold=True, align="left")
ar3 = [v(172, 386 + k * 36, 62, 28, t, fs=9, italic=True) for k, t in enumerate(["Article 1", "Article 2", "Article n"])]
enc3 = trap(258, 384, 100, 112, "Frozen\nFinBERT", fs=9)
es3 = [v(380, 386 + k * 36, 34, 28, t, fs=9) for k, t in enumerate(["e1", "e2", "en"])]
_grp3 = v(436, 376, 150, 132, "", "arcSize=8;", fill="#D5D1EA", stroke="#6B5FA8")
pa = v(448, 384, 126, 34, "Task-query\nattention", italic=True, fs=9)
pb = v(448, 428, 126, 34, "Market-to-article\ncross-attention", italic=True, fs=9)
pc = v(448, 472, 126, 30, "Scalar gate on text", italic=True, fs=9)
z3 = v(608, 424, 34, 28, "z")
xa3 = v(608, 470, 38, 28, "x, a", fs=9)
cc3 = v(668, 386, 40, 112, "Concat", fs=9)
d3 = v(730, 424, 78, 40, "Dense 16\nReLU", italic=True, fs=9)
p3 = v(824, 428, 36, 32, "P(UP)", fs=8)
grp3 = _grp3
for a in ar3: e(a, enc3)
for b in es3: e(enc3, b)
for b in es3: e(b, grp3, nx=0, ny=.5)
e(grp3, z3, ex=1, ey=.45, nx=0, ny=.5)
e(z3, cc3, nx=0, ny=.3); e(xa3, cc3, nx=0, ny=.85)
e(cc3, d3); e(d3, p3)
text(560, 358, 296, 16, "each alone, or all three combined", color="#6B5FA8", fs=9, italic=True, align="right")
e(nw, ar2[1], ex=1, ey=.08, nx=0, ny=.5); e(nw, ar3[1], ex=1, ey=.93, nx=0, ny=.5)

# ---------------- evaluation
v(885, 10, 118, 506, "", "arcSize=5;", dashed=True)
text(890, 14, 108, 18, "Evaluation", fs=10, bold=True)
s1 = v(895, 58, 98, 108, "Select on 3 expanding validation folds (purged), seeds 42, 123, 2026", fs=9)
s2 = v(895, 214, 98, 88, "Refit on all 930 development decisions", fs=9)
s3 = v(895, 382, 98, 124, "Test: 235 days, threshold 0.5\nMCC, AUC, log loss, accuracy", fs=9)
e(s1, s2); e(s2, s3)
e(p1, s1, ex=.5, ey=1, nx=0, ny=.5, pts=[(470, 147), (872, 147), (872, 112)])
e(p2, s2); e(p3, s3)

xml = '<mxGraphModel adaptiveColors="auto"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel>"
out = ROOT / "report" / "figures" / "drawio" / "pipeline.drawio"
out.write_text('<mxfile host="drawio"><diagram name="pipeline">' + xml + "</diagram></mxfile>")
print("wrote", out)
