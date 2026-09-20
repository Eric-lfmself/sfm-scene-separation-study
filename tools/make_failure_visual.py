#!/usr/bin/env python3
"""Draw three SfM failure mechanisms with original synthetic 3D geometry.

This is a deterministic scientific illustration, not reconstructed experimental
output. All buildings, image-view cards, camera centres, and links are generated
below; no dataset photographs, predicted poses, or measured point clouds are used.
Run from any directory: python tools/make_failure_visual.py
"""
from __future__ import annotations

import hashlib
import html
import json
import platform
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
SEED = 20920
BLUE, TEAL, PURPLE = '#176F9F', '#25796B', '#7053A1'
RED, INK, MUTED = '#B74938', '#172D40', '#536977'
RULE, PALE = '#D8E2E8', '#F5F8FA'
DPI = 180
DESCRIPTION = (
    'Three original synthetic 3D illustrations explain SfM failure mechanisms. '
    'Left: a stone courtyard with a tower and an industrial shed with cylindrical '
    'tanks belong to different places, but dashed red correspondences connect their '
    'camera groups into one wrong component. Centre: the same recognizable courtyard '
    'is drawn twice with blue and teal camera groups; a broken cross-session link '
    'leaves two disconnected reconstructions. Right: cameras follow a trajectory '
    'around one building; a blue reference frustum and a dashed red erroneous frustum '
    'share the same centre but face different directions. Insets enlarge the '
    'correspondence and camera-direction mechanisms. Every scene, view, point cloud '
    'and pose is schematic, not a measured result or a dataset image.'
)


def project(points, az=-47, el=24):
    """Orthographic projection of true 3D schematic coordinates to a drawing plane."""
    p = np.asarray(points, dtype=float)
    az, el = np.radians([az, el])
    horizontal = np.array([np.cos(az), -np.sin(az), 0.])
    vertical = np.array([-np.sin(el) * np.sin(az), -np.sin(el) * np.cos(az), np.cos(el)])
    depth = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    return np.column_stack((p @ horizontal, p @ vertical)), p @ depth


def point_surface(rng, a, b, c, color, count=180):
    a, b, c = map(np.asarray, (a, b, c))
    uv = rng.random((count, 2))
    points = a + uv[:, :1] * (b - a) + uv[:, 1:] * (c - a)
    colors = np.clip(np.asarray(to_rgb(color))[None] + rng.normal(0, .028, (count, 1)), 0, 1)
    return points, colors


def architecture(kind='courtyard', seed=SEED):
    """Return colored point samples, native polygon faces, and a ground footprint."""
    rng = np.random.default_rng(seed)
    points, colors, faces = [], [], []

    def quad(a, b, c, color, count=180):
        p, rgb = point_surface(rng, a, b, c, color, count)
        points.append(p); colors.append(rgb)
        a, b, c = map(np.asarray, (a, b, c))
        faces.append((np.array([a, b, b + c - a, c]), color))

    def block(x, y, w, d, h, color='#AD9475'):
        quad([x, y, 0], [x+w, y, 0], [x, y, h], color, 300)
        quad([x+w, y, 0], [x+w, y+d, 0], [x+w, y, h], '#8E785F', 220)
        quad([x, y+d, 0], [x+w, y+d, 0], [x, y+d, h], '#C0AA8E', 180)
        quad([x, y, h], [x+w, y, h], [x, y+d, h], '#B5AB98', 180)

    def window(x, y, z, w=.22, h=.31):
        quad([x, y-.006, z], [x+w, y-.006, z], [x, y-.006, z+h], '#416379', 32)
        quad([x+.03, y-.012, z+.04], [x+w*.5, y-.012, z+.04],
             [x+.03, y-.012, z+h-.04], '#A4BECC', 16)

    if kind == 'courtyard':
        block(-1.6, .25, 3.2, .78, 1.22)
        block(-1.62, -1.0, .72, 1.30, 1.22, '#BBA385')
        # Tall square tower and a four-sided pointed roof make the site recognizable.
        block(-1.12, .15, .65, .72, 2.12, '#B39A7B')
        for a, b in [([-.47, .15, 2.12], [-1.12, .15, 2.12]),
                     ([-.47, .87, 2.12], [-.47, .15, 2.12]),
                     ([-1.12, .87, 2.12], [-.47, .87, 2.12]),
                     ([-1.12, .15, 2.12], [-1.12, .87, 2.12])]:
            apex = np.array([-.795, .51, 2.65]); a, b = np.array(a), np.array(b)
            uv = rng.random((130, 2)); uv[uv.sum(1)>1] = 1-uv[uv.sum(1)>1]
            points.append(a + uv[:, :1]*(b-a) + uv[:, 1:]*(apex-a))
            colors.append(np.tile(to_rgb('#796C63'), (len(uv), 1)))
            faces.append((np.array([a,b,apex]), '#796C63'))
        for x in [-.19, .34, .87, 1.34]:
            window(x, .25, .62, .18, .31)
        window(-.98, .15, 1.53, .31, .38)
        # Semicircular arch: the dark recess and tan rim are geometry, not photo texture.
        th = np.linspace(0, np.pi, 90)
        arch = np.column_stack((.57+.24*np.cos(th), np.full_like(th,.236), .41+.24*np.sin(th)))
        points.append(arch); colors.append(np.tile(to_rgb('#6C665C'), (len(th),1)))
        quad([.33,.239,0], [.81,.239,0], [.33,.239,.40], '#6C665C', 85)
        # Paving strokes reinforce a courtyard footprint.
        for yy in np.linspace(-1.04,.20,7):
            quad([-1.58,yy,0], [1.6,yy,0], [-1.58,yy+.012,0], '#B2B6AB', 50)
        footprint = np.array([[-1.9,-1.3,0],[1.9,-1.3,0],[1.9,1.2,0],[-1.9,1.2,0]])
    else:
        block(-1.72, .12, 3.44, .91, 1.04, '#BCAD8A')
        # Three sawtooth roof bays, unmistakably distinct from the courtyard tower.
        for x in [-1.72,-.58,.56]:
            quad([x,.12,1.04],[x+.74,.12,1.48],[x,1.03,1.04],'#918D7D',160)
            quad([x+.74,.12,1.48],[x+1.14,.12,1.04],[x+.74,1.03,1.48],'#707E7D',100)
        for x in np.linspace(-1.45,1.20,7):
            window(x,.12,.42,.20,.35)
        # Cylindrical tanks are point-sampled closed surfaces.
        for x in [-.88,.05,.98]:
            theta = rng.uniform(0,2*np.pi,420); zz = rng.uniform(0,.80,420)
            p=np.column_stack((x+.24*np.cos(theta), -.64+.24*np.sin(theta), zz))
            points.append(p)
            shade=.72+.18*np.cos(theta-.5)
            colors.append(np.clip(np.asarray(to_rgb('#9A9D87'))[None]*shade[:,None],0,1))
            rr=.24*np.sqrt(rng.random(100)); th=rng.uniform(0,2*np.pi,100)
            points.append(np.column_stack((x+rr*np.cos(th),-.64+rr*np.sin(th),np.full(100,.80))))
            colors.append(np.tile(to_rgb('#BEC2AC'),(100,1)))
        footprint=np.array([[-2.0,-1.2,0],[2.0,-1.2,0],[2.0,1.3,0],[-2.0,1.3,0]])
    return np.concatenate(points), np.concatenate(colors), faces, footprint


def transform(points, origin, scale, az=-47, el=24):
    xy, _ = project(points,az,el)
    return np.asarray(origin) + scale*xy


def draw_scene(ax, kind, origin, scale, accent, seed=SEED, az=-47, alpha=1., pointsize=1.8):
    pts, colors, faces, footprint = architecture(kind,seed)
    floor=transform(footprint,origin,scale,az)
    ax.add_patch(Polygon(floor,facecolor=accent,alpha=.055,edgecolor=accent,lw=.8,zorder=1))
    for values in np.linspace(-1.65,1.65,7):
        xy=transform([[values,-1.2,-.015],[values,1.2,-.015]],origin,scale,az)
        ax.plot(*xy.T,color=RULE,lw=.4,zorder=1)
    for face, color in sorted(faces,key=lambda f:project(f[0],az)[1].mean()):
        ax.add_patch(Polygon(transform(face,origin,scale,az),facecolor=color,
                             edgecolor=color,linewidth=.35,alpha=.09*alpha,zorder=2))
    xy, depth=project(pts,az)
    order=np.argsort(depth)
    xy=np.asarray(origin)+scale*xy
    ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=pointsize,
               alpha=.84*alpha,linewidths=0,zorder=3,rasterized=False)


def frustum(ax, centre, target, origin, scale, color, size=.37, az=-47,
            dashed=False, fill=.045, linewidth=.8, zorder=5):
    centre=np.asarray(centre,float); target=np.asarray(target,float)
    direction=target-centre; direction/=np.linalg.norm(direction)
    right=np.cross(direction,[0,0,1.]); right/=np.linalg.norm(right)
    up=np.cross(right,direction)
    plane=centre+size*direction
    corners=np.array([plane+.47*size*sx*right+.32*size*sy*up for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    pc=transform([centre],origin,scale,az)[0]
    xy=transform(corners,origin,scale,az)
    ax.add_patch(Polygon(xy,facecolor=color,edgecolor='none',alpha=fill,zorder=zorder))
    style=(0,(3,2)) if dashed else '-'
    for q in xy:
        ax.plot([pc[0],q[0]],[pc[1],q[1]],color=color,lw=linewidth,ls=style,zorder=zorder)
    loop=np.vstack((xy,xy[0]))
    ax.plot(*loop.T,color=color,lw=linewidth,ls=style,zorder=zorder)
    ax.scatter([pc[0]],[pc[1]],s=8,c=color,zorder=zorder+1)
    return pc


def cameras(ax, origin, scale, color, angles=(-145,-115,-85,-55,-25), radius=2.55, az=-47):
    theta=np.radians(angles)
    centers=np.column_stack((radius*np.cos(theta), radius*.70*np.sin(theta), np.full(len(theta),.55)))
    xy=transform(centers,origin,scale,az)
    ax.plot(*xy.T,color=color,lw=1,alpha=.58,zorder=4)
    for c in centers:
        frustum(ax,c,[0,0,.7],origin,scale,color,size=.49,az=az)
    return centers,xy


def text(ax,x,y,content,size=11,color=INK,weight='normal',ha='left',**kwargs):
    return ax.text(x,y,content,fontsize=size,color=color,fontweight=weight,ha=ha,
                   va='center',linespacing=1.35,**kwargs)


def arrow(ax,start,end,color=MUTED,ls='-',connection='arc3',width=1.2):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,
                                lw=width,color=color,linestyle=ls,
                                connectionstyle=connection,zorder=8))


def card(ax,x,y,w,h,label,accent,kind='courtyard',az=-47):
    frame=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',
                        facecolor='#F5F7F7',edgecolor=RULE,lw=.85,zorder=1)
    ax.add_patch(frame)
    before=set(ax.get_children())
    draw_scene(ax,kind,(x+w*.5,y+h*.32),w/5.3,accent,az=az,pointsize=.48)
    for artist in set(ax.get_children())-before:
        artist.set_clip_path(frame)
    text(ax,x+w*.5,y-2.4,label,8.2,color=accent,weight='bold',ha='center')



def compact_native_svg(svg):
    """Deduplicate identical clipping geometry without changing any plotted point.

    Matplotlib emits a separate rounded-thumbnail clip path for thousands of
    markers. Reuse these definitions, hoist identical clipping/opacity to their
    collection, and remove formatting whitespace. Coordinates and point order
    are preserved exactly; this is SVG structure compression, not resampling.
    """
    namespaces = {
        '': 'http://www.w3.org/2000/svg',
        'xlink': 'http://www.w3.org/1999/xlink',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'cc': 'http://creativecommons.org/ns#',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    }
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(svg)
    tag = lambda name: '{' + namespaces[''] + '}' + name
    def structural_key(element):
        return (element.tag, tuple(sorted((k, v) for k, v in element.attrib.items() if k != 'id')),
                tuple(structural_key(child) for child in element))
    canonical, remap = {}, {}
    for parent in root.iter():
        for child in list(parent):
            if child.tag != tag('clipPath'):
                continue
            key = structural_key(child)
            if key in canonical:
                remap[child.attrib['id']] = canonical[key]
                parent.remove(child)
            else:
                canonical[key] = child.attrib['id']
    for element in root.iter():
        for attr, value in list(element.attrib.items()):
            if value.startswith('url(#') and value.endswith(')'):
                old_id = value[5:-1]
                if old_id in remap:
                    element.set(attr, 'url(#' + remap[old_id] + ')')
    for parent in root.iter():
        if not parent.get('id', '').startswith('PathCollection_'):
            continue
        wrappers = [child for child in parent if child.tag != tag('defs')]
        if wrappers and all(child.tag == tag('g') and set(child.attrib) == {'clip-path'}
                            and len(child) == 1 and child[0].tag == tag('use') for child in wrappers):
            clips = {child.get('clip-path') for child in wrappers}
            if len(clips) == 1:
                parent.set('clip-path', clips.pop())
                for wrapper in wrappers:
                    index = list(parent).index(wrapper)
                    parent.remove(wrapper)
                    parent.insert(index, wrapper[0])
        uses = [child for child in parent if child.tag == tag('use')]
        styles = [dict(item.strip().split(':', 1) for item in child.get('style', '').split(';')
                       if ':' in item) for child in uses]
        opacity = {style.get('fill-opacity', '').strip() for style in styles}
        if uses and len(opacity) == 1 and '' not in opacity:
            parent.set('fill-opacity', opacity.pop())
            for child, style in zip(uses, styles):
                style.pop('fill-opacity')
                if set(style) == {'fill'}:
                    child.set('fill', style['fill'].strip())
                    child.attrib.pop('style', None)
                else:
                    child.set('style', ';'.join(k + ':' + v for k, v in style.items()))
    # No formatting whitespace is part of a text label; its text node stays intact.
    serialized = ET.tostring(root, encoding='unicode')
    return '<?xml version="1.0" encoding="utf-8"?>\n' + re.sub(r'>\s+<', '><', serialized) + '\n'


def draw():
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'sfm-failure-visual-v1',
                         'font.size':11,'axes.linewidth':.8}):
        fig=plt.figure(figsize=(19.5,10.1),facecolor='white')
        fig.text(.034,.948,'RECONSTRUCTION DIAGNOSTICS',fontsize=10.8,color=BLUE,weight='bold')
        fig.text(.034,.906,'Three ways a plausible 3D model can be wrong',fontsize=25,weight='bold',color=INK)
        fig.text(.034,.866,'I inspect which images connect, which sessions separate, and where cameras point.',
                 fontsize=12.4,color=MUTED)
        axes=[]
        for i in range(3):
            ax=fig.add_axes([.031+i*.324,.111,.305,.701])
            ax.set(xlim=(0,100),ylim=(0,105));ax.axis('off');axes.append(ax)
        a,b,c=axes
        for ax,kicker,title,subtitle,col in [
            (a,'01 / FALSE MERGE','Different places, one model','A false bridge can join unrelated scenes.',BLUE),
            (b,'02 / FALSE SPLIT','One place, two models','A missing bridge can separate revisit sessions.',TEAL),
            (c,'03 / POSE MISMATCH','Right centres, wrong directions','Registration does not guarantee correct orientation.',PURPLE)]:
            text(ax,1,101,kicker,10.2,col,'bold')
            text(ax,1,94,title,16.5,INK,'bold')
            text(ax,1,88,subtitle,9.7,MUTED)
            ax.plot([1,98],[84,84],color=RULE,lw=.9)
        # A: two visibly different physical sites, placed independently for illustration.
        draw_scene(a,'courtyard',(26,60),9.4,BLUE,pointsize=2.0)
        _, ca=cameras(a,(26,60),9.4,BLUE,angles=(-150,-115,-80,-45),radius=2.50)
        text(a,2,81,'SITE A · tower + courtyard',9,BLUE,'bold')
        draw_scene(a,'shed',(76,47),9.5,TEAL,seed=SEED+1,pointsize=2.0)
        _, cb=cameras(a,(76,47),9.5,TEAL,angles=(-150,-115,-80,-45),radius=2.50)
        text(a,58,66,'SITE B · shed + tanks',9,TEAL,'bold')
        for start,end in zip(ca[:3],cb[:3]):
            a.plot([start[0],end[0]],[start[1],end[1]],color=RED,lw=1.6,ls=(0,(4,3)),zorder=7)
        text(a,41,55,'false links',9.7,RED,'bold',ha='center',
             bbox=dict(facecolor='white',edgecolor='none',pad=2.3))
        # Illustrative view cards: repeated facade patterns do not imply one place.
        card(a,6,17,36,18,'view from site A',BLUE,'courtyard')
        card(a,58,17,36,18,'view from site B',TEAL,'shed')
        for yy in [24,28,31]:
            a.plot([33,69],[yy,yy-1],color=RED,ls=(0,(3,2)),lw=.85,alpha=.8,zorder=7)
        text(a,50,9,'Local appearance can agree\nwhile scene identity is wrong.',10.3,INK,ha='center')
        # B: identical geometry, separate labels, and no supporting cross-session edge.
        draw_scene(b,'courtyard',(26,60),9.4,BLUE,pointsize=2.0)
        _, ba=cameras(b,(26,60),9.4,BLUE,angles=(-150,-115,-80,-45),radius=2.50)
        text(b,2,81,'SESSION A',9,BLUE,'bold')
        draw_scene(b,'courtyard',(75,46.5),9.4,TEAL,pointsize=2.0)
        _, bb=cameras(b,(75,46.5),9.4,TEAL,angles=(-150,-115,-80,-45),radius=2.50)
        text(b,61,70,'SESSION B',9,TEAL,'bold')
        b.plot([31,43],[51,47],color=MUTED,lw=1.1,ls=(0,(3,3)),zorder=7)
        b.plot([52,64],[44,40],color=MUTED,lw=1.1,ls=(0,(3,3)),zorder=7)
        b.plot([44,50],[45,49],color=RED,lw=2,zorder=8)
        b.plot([44,50],[49,45],color=RED,lw=2,zorder=8)
        text(b,46,54,'no verified bridge',9.2,RED,'bold',ha='center',
             bbox=dict(facecolor='white',edgecolor='none',pad=2.3))
        card(b,6,17,36,18,'session A view',BLUE,'courtyard',az=-60)
        card(b,58,17,36,18,'session B view',TEAL,'courtyard',az=-23)
        text(b,50,9,'The tower and courtyard recur.\nSeparate models still miss the revisit.',10.3,INK,ha='center')
        # C: same physical centre, visibly divergent camera frusta.
        draw_scene(c,'courtyard',(53,54),12.7,PURPLE,pointsize=2.3)
        centers,xy=cameras(c,(53,54),12.7,BLUE,angles=(-150,-122,-94,-66,-38,-10),radius=2.35)
        wrong=centers[2]
        away=wrong+(wrong-np.array([0.,0.,.55]))
        pc=frustum(c,wrong,away,(53,54),12.7,RED,size=.85,dashed=True,fill=.07,linewidth=1.6,zorder=8)
        text(c,3,82,'one building · registered trajectory',9,PURPLE,'bold')
        text(c,3,42,'misoriented pose',9.5,RED,'bold')
        arrow(c,(27,42),pc,RED,connection='arc3,rad=.18')
        # Large orientation inset, anchored to a shared camera centre.
        c.add_patch(FancyBboxPatch((7,15),88,21,boxstyle='round,pad=0,rounding_size=1.8',
                                   facecolor=PALE,edgecolor=RULE,lw=.8,zorder=1))
        text(c,10,33,'SAME CENTRE, DIFFERENT ROTATION',8.7,MUTED,'bold')
        origin=(48,23)
        frustum(c,[0,0,0],[1,0,0],origin,9.5,BLUE,size=1.5,az=0,fill=.07,linewidth=1.2)
        frustum(c,[0,0,0],[-1,0,0],origin,9.5,RED,size=1.5,az=0,dashed=True,fill=.07,linewidth=1.2)
        c.scatter([48],[23],c=INK,s=20,zorder=9)
        arrow(c,(48,23),(72,23),BLUE,width=1.3)
        arrow(c,(48,23),(24,23),RED,ls=(0,(3,2)),width=1.3)
        text(c,19,17,'erroneous',8.7,RED,ha='center')
        text(c,77,17,'reference',8.7,BLUE,ha='center')
        text(c,50,9,'Check rotation after alignment.\nCamera coverage alone can hide the error.',10.3,INK,ha='center')
        # Sparse dividers and a single explicit evidence boundary.
        for x in [.347,.671]:
            fig.add_artist(plt.Line2D([x,x],[.12,.82],transform=fig.transFigure,color=RULE,lw=.9))
        fig.text(.034,.065,'SCHEMATIC · All geometry, point clouds, view cards and camera poses are synthetic.',
                 fontsize=10.1,color=MUTED)
        fig.text(.034,.035,'No dataset photographs or measured reconstructions are shown. Dashed red marks indicate the illustrated error.',
                 fontsize=10.1,color=MUTED)
        OUT.mkdir(exist_ok=True)
        name='failure-visual'
        fig.savefig(OUT/f'{name}.png',dpi=DPI,facecolor='white',transparent=False)
        fig.savefig(OUT/f'{name}.svg',facecolor='white',transparent=False,
                    metadata={'Date':None,'Creator':'tools/make_failure_visual.py','Description':DESCRIPTION})
        svgpath=OUT/f'{name}.svg';svg=svgpath.read_text()
        svg=svg.replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, Helvetica, sans-serif")
        offset=svg.index('>',svg.index('<svg'))+1
        svg=svg[:offset]+'\n<title>Three SfM failure mechanisms</title>\n<desc>'+html.escape(DESCRIPTION)+'</desc>'+svg[offset:]
        svgpath.write_text(compact_native_svg(svg));ET.parse(svgpath)
        manifest=dict(name=name,description=DESCRIPTION,files=[f'{name}.svg',f'{name}.png'],
                      width_inches=19.5,height_inches=10.1,png_dpi=DPI,background='#FFFFFF',
                      svg_text='editable, fonts not embedded',
                      svg_font_fallback=['DejaVu Sans','Arial','Helvetica','sans-serif'],
                      png_font='DejaVu Sans (rendered by Matplotlib)',
                      uncertainty='Not applicable: original synthetic schematic, with no measured estimates.',
                      software=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__),
                      sources=[],
                      transformations=['Original parameterized 3D buildings and camera frusta projected orthographically.',
                                       'Point-cloud texture uses a fixed random seed; view cards reuse the same synthetic geometry.',
                                       'No measured coordinates, images or matching results are used.',
                                       'SVG reuses identical clip paths and collection styling; every point coordinate is retained.'],
                      evidence_status='schematic',geometry_seed=SEED,
                      generator='tools/make_failure_visual.py',
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (OUT/f'{name}.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        plt.close(fig)
        print(f'Wrote {name}.svg, {name}.png, and {name}.manifest.json')


if __name__=='__main__':
    draw()
