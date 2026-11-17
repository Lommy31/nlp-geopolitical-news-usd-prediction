"""Generate figures/drawio/architecture.drawio (editable, draw.io paper style).
Run from repo root: python3 report/eda-visualization/make_architecture_drawio.py
Export: xvfb-run -a drawio -x -f pdf --crop --no-sandbox -o report/figures/assets/model_architecture.pdf report/figures/drawio/architecture.drawio
Content verified against src/task_3/training/{models,attention,data}.py (improved_v1)."""
import base64, io
from pathlib import Path
from xml.sax.saxutils import quoteattr
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ICON = ROOT / "report" / "figures" / "icons"
cells, n = [], [1]
def nid():
    n[0] += 1; return f"c{n[0]}"

def png_uri(name, size=160):
    im = Image.open(ICON / name).convert("RGBA").resize((size, size), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return "data:image/png," + base64.b64encode(b.getvalue()).decode()

def svg_uri(svg):
    return "data:image/svg+xml," + base64.b64encode(svg.encode()).decode()

FLAME = ICON / "flame.png"
FLAME_URI = png_uri("flame.png") if FLAME.exists() else svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#FF9A0A"/><stop offset="1" stop-color="#FF5A14"/></linearGradient></defs>'
    '<path d="M38 4 C34 20 44 28 36 42 C30 52 22 60 24 74 C26 90 40 97 52 97 C70 97 84 84 82 66 C81 52 72 44 66 32 C64 40 60 42 56 40 C60 28 50 14 38 4Z" fill="url(#g)"/>'
    '<path d="M50 52 C56 62 62 66 62 78 C62 88 56 94 50 94 C42 94 36 88 36 80 C36 70 46 64 50 52Z" fill="#FB9B05"/>'
    '<path d="M50 70 C54 76 56 80 55 86 C54 91 46 92 45 86 C44 80 48 76 50 70Z" fill="#FFE000"/></svg>')
CHART_URI = svg_uri('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80"><path d="M8 4V72H96" fill="none" stroke="#111" stroke-width="5"/>'
                    '<path d="M14 58 L26 40 L36 50 L50 24 L60 36 L72 16 L84 30 L94 22" fill="none" stroke="#3775BA" stroke-width="5" stroke-linejoin="round"/></svg>')
SNOW_URI, FOLDER_URI = png_uri("snowflake.png"), png_uri("folder_docs.png")

F = "fontFamily=Helvetica;html=1;whiteSpace=wrap;"
def v(x, y, w, h, t="", fs=11, fill="#FFFFFF", stroke="#000000", sw=1, rnd=True, arc=14, italic=False, bold=False, dashed=False, extra="", color="#000000", id=None):
    i = id or nid()
    st = (2 if italic else 0) + (1 if bold else 0)
    s = F + f"fontSize={fs};fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={color};"
    if rnd: s += f"rounded=1;arcSize={arc};"
    if st: s += f"fontStyle={st};"
    if dashed: s += "dashed=1;"
    s += extra
    cells.append(f'<mxCell id="{i}" value={quoteattr(t)} style="{s}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def txt(x, y, w, h, t, fs=10, color="#000000", italic=False, bold=False, align="center"):
    i = nid(); st = (2 if italic else 0) + (1 if bold else 0)
    s = F + f"text;fontSize={fs};fontColor={color};align={align};verticalAlign=middle;strokeColor=none;fillColor=none;" + (f"fontStyle={st};" if st else "")
    cells.append(f'<mxCell id="{i}" value={quoteattr(t)} style="{s}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def img(x, y, w, h, uri):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="" style="shape=image;imageAspect=1;aspect=fixed;image={uri};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def trap(x, y, w, h, t, fill, stroke, fs=12):
    i = nid()
    s = F + f"shape=trapezoid;perimeter=trapezoidPerimeter;direction=south;fixedSize=1;size=30;fontSize={fs};fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
    cells.append(f'<mxCell id="{i}" value={quoteattr(t)} style="{s}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def vec(x, y, w=120, h=30, fill="#DCE8F6", stroke="#1F4E8C", k=5, dash=False):
    r = v(x, y, w, h, "", fill=fill, stroke=stroke, sw=2, arc=30, dashed=dash, extra="dashPattern=3 3;" if dash else "")
    d = h - 10; step = (w - 12) / k
    for j in range(k):
        v(x + 8 + j * step + (step - d) / 2, y + 5, d, d, "", rnd=False, extra="ellipse;", sw=1)
    return r

def e(a, b, dashed=False, color="#000000", ex=None, ey=None, nx=None, ny=None, pts=None, label="", sw=1.5):
    i = nid()
    s = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth={sw};fontSize=10;fontFamily=Helvetica;strokeColor={color};"
    if dashed: s += "dashed=1;"
    for kx, val in (("exitX", ex), ("exitY", ey), ("entryX", nx), ("entryY", ny)):
        if val is not None: s += f"{kx}={val};"
    g = '<mxGeometry relative="1" as="geometry"/>' if not pts else '<mxGeometry relative="1" as="geometry"><Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + "</Array></mxGeometry>"
    cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style="{s}" edge="1" parent="1" source="{a}" target="{b}">{g}</mxCell>')

def ue(x1, y1, x2, y2, dashed=False, color="#000000"):
    i = nid()
    st = "edgeStyle=none;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=%s;" % color + ("dashed=1;" if dashed else "")
    cells.append(f'<mxCell id="{i}" value="" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

def ln(x1, y1, x2, y2, arrow=False, dashed=False, color="#000000"):
    i = nid()
    st = "edgeStyle=none;html=1;endFill=1;strokeWidth=1.5;strokeColor=%s;" % color + ("endArrow=block;" if arrow else "endArrow=none;") + ("dashed=1;" if dashed else "")
    cells.append(f'<mxCell id="{i}" value="" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

def panel(x, y, w, h, title, fill="#FFFFFF"):
    v(x, y, w, h, "", fill=fill, stroke="#000000", sw=1.5, rnd=True, arc=3, dashed=True, extra="dashPattern=6 4;")
    tw = max(150, 9 * len(title) + 40)
    v(x + (w - tw) / 2, y - 16, tw, 32, title, fs=14, fill="#F0F0F0", stroke="#000000", sw=1.5, rnd=False, bold=False)

# =============================== compact layout (canvas ~835 px wide) ===============================
GREY, DOT = "#8C8C8C", "dashPattern=2 3;"
OPT = 'dashed=True, extra="dashPattern=3 3;"'
def frame(x, y, w, h, title):
    v(x, y, w, h, "", fill="#FFFFFF", stroke=GREY, sw=1.3, rnd=True, arc=3, dashed=True, extra=DOT)
    tw = max(110, int(7.4 * len(title)) + 26)
    v(x + (w - tw) / 2, y - 12, tw, 24, title, fs=12, fill="#F0F0F0", stroke="#000000", sw=1.3, rnd=False)
def seg(x, y, w, h, t, dash=False):
    v(x, y, w, h, t, fs=11, fill="#DCE8F6", stroke="#1F4E8C", sw=2, arc=10, dashed=dash, extra="dashPattern=3 3;" if dash else "")
def vc(x, y, w, h, fill, stroke, k, dash=False):          # circle-strip vector
    v(x, y, w, h, "", fill=fill, stroke=stroke, sw=2, arc=30, dashed=dash, extra="dashPattern=3 3;" if dash else "")
    d = h - 10; step = (w - 12) / k
    for j in range(k):
        v(x + 8 + j * step + (step - d) / 2, y + 5, d, d, "", rnd=False, extra="ellipse;", sw=1)

T, B = 28, 312
# ---- left inputs
img(12, 73, 44, 44, FOLDER_URI); txt(0, 120, 72, 44, "Day-t news\n(title + first\n180 words)", fs=10)
img(12, 252, 46, 40, CHART_URI); txt(0, 294, 72, 30, "Market series\n(14 features)", fs=10)

# ---- panel 1
frame(78, T, 240, B - T, "Feature Extraction")
trap(88, 50, 80, 90, "Frozen\nencoder", "#DCE8F6", "#1F4E8C", fs=11)
img(92, 54, 24, 24, SNOW_URI)
vc(196, 54, 100, 22, "#DCE8F6", "#1F4E8C", 5); vc(196, 84, 100, 22, "#DCE8F6", "#1F4E8C", 5); vc(196, 114, 100, 22, "#DCE8F6", "#1F4E8C", 5)
txt(190, 138, 112, 18, "article vectors e<sub>i</sub>", fs=10, italic=True)
v(88, 255, 70, 34, "Standard\nScaler", fs=10)
vc(196, 261, 100, 22, "#F3D6D5", "#B64342", 5); txt(196, 241, 100, 18, "x (14)", fs=10)

# ---- panel 2
frame(336, T, 230, B - T, "Level 3 Text Slots")
mean = v(350, 50, 100, 38, "Mean pooling", fs=11, fill="#EDEDED", stroke="#7A7A7A", sw=2)
tq = v(350, 118, 100, 56, "Task-query\nattention", fs=11, fill="#DCE8F6", stroke="#3775BA", sw=2, **{"dashed": True, "extra": "dashPattern=3 3;spacingTop=12;"})
ca = v(350, 229, 100, 66, "Market-to-article\ncross-attention", fs=11, fill="#F3D6D5", stroke="#B64342", sw=2, **{"dashed": True, "extra": "dashPattern=3 3;spacingTop=12;"})
img(354, 121, 16, 16, FLAME_URI); img(354, 232, 16, 16, FLAME_URI)
vc(366, 190, 68, 16, "#FFFFFF", "#3775BA", 3); txt(436, 188, 22, 18, "q", fs=10, italic=True)    # learned query vector
vc(466, 58, 90, 22, "#DCE8F6", "#1F4E8C", 4); vc(466, 135, 90, 22, "#DCE8F6", "#1F4E8C", 4, dash=True); vc(466, 251, 90, 22, "#DCE8F6", "#1F4E8C", 4, dash=True)
txt(466, 42, 90, 16, "s<sub>0</sub>", fs=10); txt(466, 119, 90, 16, "s<sub>1</sub>", fs=10); txt(466, 235, 90, 16, "s<sub>2</sub>", fs=10)

# ---- panel 3
frame(600, T, 230, B - T, "Gated Fusion & Head")
for k_, (nm_, d_) in enumerate([("s<sub>0</sub>", False), ("s<sub>1</sub>", True), ("s<sub>2</sub>", True)]):
    seg(612 + k_ * 40, 58, 36, 22, nm_, dash=d_)
v(610, 94, 150, 80, "", fill="#FBF6E6", stroke="#000000", sw=1.5, arc=8, dashed=True, extra="dashPattern=3 3;")
img(614, 97, 18, 18, FLAME_URI); txt(636, 97, 50, 18, "Gate", fs=11, bold=True, align="left")
v(618, 118, 134, 20, "Sigmoid", fs=10, fill="#DDEBCB", stroke="#7BA55A", sw=1.5)
v(618, 148, 134, 20, "Linear", fs=10, fill="#DCE8F6", stroke="#3775BA", sw=1.5)
v(782, 114, 26, 26, "", rnd=False, extra="ellipse;", sw=1.5); v(791, 123, 8, 8, "", rnd=False, extra="ellipse;", fill="#000000", sw=1)
vc(612, 190, 66, 18, "#F3D6D5", "#B64342", 5); txt(680, 189, 40, 20, "x (14)", fs=9, italic=True, align="left")
vc(722, 190, 40, 18, "#D3E8EA", "#42949E", 2); txt(764, 189, 40, 20, "a (2)", fs=9, italic=True, align="left")
v(610, 228, 210, 24, "Concat", fs=11, sw=1.5)
v(612, 268, 108, 34, "Linear 16\nReLU · Dropout", fs=10, fill="#FBEBC4", stroke="#C9A227", sw=2); img(616, 271, 16, 16, FLAME_URI)
v(732, 270, 88, 30, "Linear 1\nσ → P(UP)", fs=9, fill="#DCE8F6", stroke="#1F4E8C", sw=2, bold=True)

# ---- connectors (straight segments, drawn after frames so they sit on top)
ln(58, 95, 88, 95, arrow=True)                                   # news -> encoder
ln(168, 95, 196, 95, arrow=True)                                 # encoder -> vectors
ln(62, 272, 88, 272, arrow=True); ln(158, 272, 196, 272, arrow=True)   # market -> scaler -> x
for yy in (65, 95, 125): ln(296, yy, 304, yy)                    # all article vectors -> bus
ln(304, 65, 304, 125); ln(304, 95, 327, 95); ln(327, 69, 327, 247)
for yy in (69, 146, 247): ln(327, yy, 350, yy, arrow=True)       # bus -> three slot boxes
ln(296, 272, 350, 272, arrow=True, dashed=True, color="#B64342"); txt(298, 250, 44, 14, "query", fs=9, italic=True, color="#B64342")
ln(400, 190, 400, 174, arrow=True)                               # learned q -> task-query
for yy in (69, 146, 262): ln(450, yy, 466, yy, arrow=True)       # boxes -> slot vectors
for yy in (69, 146, 262): ln(556, yy, 583, yy)                   # slot vectors -> bus
ln(583, 69, 583, 262); ln(583, 69, 612, 69, arrow=True)          # bus -> concatenated slots
ln(728, 69, 795, 69); ln(795, 69, 795, 114, arrow=True)          # slots -> multiply
ln(752, 128, 782, 128, arrow=True); txt(758, 110, 20, 14, "g", fs=10, italic=True)   # sigmoid -> multiply
ln(685, 148, 685, 138, arrow=True)                               # linear -> sigmoid
ln(645, 190, 645, 168, arrow=True, dashed=True, color="#B64342")    # x -> gate
ln(737, 190, 737, 168, arrow=True, dashed=True, color="#42949E")    # log n -> gate
ln(645, 208, 645, 228, arrow=True, color="#B64342"); ln(742, 208, 742, 228, arrow=True, color="#42949E")   # x, a -> concat
ln(795, 140, 795, 228, arrow=True)                               # g * slots -> concat
ln(665, 252, 665, 268, arrow=True); ln(720, 285, 732, 285, arrow=True)

xml = '<mxGraphModel adaptiveColors="auto"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel>"
out = ROOT / "report" / "figures" / "drawio" / "architecture.drawio"
out.write_text('<mxfile host="drawio"><diagram name="architecture">' + xml + "</diagram></mxfile>")
print("wrote", out, len(xml) // 1024, "KB")
