#!/usr/bin/env python3
"""Original scientific illustration from deterministic, projected 3D geometry.

This is an explanatory diagram, not a rendering of an experimental reconstruction.
Run with the dependencies in requirements-figures.txt. No external images or network.
"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
SEED=20260920
INK='#183449'; MUTED='#526878'; RULE='#D8E2E8'
BLUE='#176F9F'; TEAL='#25796B'; VIOLET='#7053A1'; ORANGE='#B95135'
PALES=['#EAF3F9','#EAF4F0','#F1EEF8']
VIEWS=[np.array([7.8,9.,5.7]),np.array([5.4,10.8,4.7]),np.array([9.,6.8,4.9])]
TARGET=np.array([0.,.5,1.])


def text(ax,x,y,s,size=11,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,
                   linespacing=1.4,**kw)


def box(ax,x,y,w,h,face='#FFFFFF',edge=RULE,r=1.2,lw=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=face,edgecolor=edge,linewidth=lw)
    ax.add_patch(p);return p


def arrow(ax,a,b,c=MUTED,lw=1.5,style='-|>',**kw):
    p=FancyArrowPatch(a,b,color=c,arrowstyle=style,mutation_scale=12,linewidth=lw,**kw)
    ax.add_patch(p)


def project(points,position,target=TARGET):
    points=np.asarray(points,float); forward=np.asarray(target)-position
    forward=forward/np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    delta=points-position
    depth=delta@forward
    return np.column_stack((delta@right,delta@up))/depth[:,None],depth


def geometry():
    """Warm L-shaped courtyard architecture, windows and ground features."""
    faces=[];lines=[]
    def face(v,c,edge=None):faces.append((np.array(v,float),c,edge or '#B8A78B'))
    # Ground and courtyard paving: all coordinates are in a shared synthetic 3D frame.
    face([[-5,-3,-.02],[5,-3,-.02],[5,6,-.02],[-5,6,-.02]],'#E2D7BC','#CFC3A9')
    for x in np.arange(-1.35,5,.65):lines.append(([[x,.15,0],[x,5.5,0]],'#CABFA9',.35))
    for y in np.arange(.15,6,.65):lines.append(([[-1.35,y,0],[4.5,y,0]],'#CABFA9',.35))
    def block(x0,x1,y0,y1,h):
        face([[x0,y0,0],[x0,y1,0],[x0,y1,h],[x0,y0,h]],'#D0B991')
        face([[x1,y0,0],[x1,y1,0],[x1,y1,h],[x1,y0,h]],'#D4BF9C')
        face([[x0,y0,0],[x1,y0,0],[x1,y0,h],[x0,y0,h]],'#D3B997')
        face([[x0,y1,0],[x1,y1,0],[x1,y1,h],[x0,y1,h]],'#F0DEB8')
        face([[x0,y0,h],[x1,y0,h],[x1,y1,h],[x0,y1,h]],'#C19A76')
        # Roof edging / cornice, visible but not falsely textured.
        lines.append(([[x0,y1,h+.015],[x1,y1,h+.015]],'#967653',1.1))
    block(-3.2,3.2,-1.6,0,3.3)
    block(-3.2,-1.6,0,3.2,2.65)
    # Main courtyard-facing facade: repeated windows with frames and mullions.
    for x in [-2.6,-1.5,-.4,.7,1.8,2.9]:
        for z in [.62,1.58,2.54]:
            w=.6;h=.62;y=.025
            face([[x-w/2,y,z-h/2],[x+w/2,y,z-h/2],[x+w/2,y,z+h/2],[x-w/2,y,z+h/2]],'#678696','#8F785E')
            lines.append(([[x,y+.015,z-h/2],[x,y+.015,z+h/2]],'#D7E4E7',.55))
            lines.append(([[x-w/2,y+.015,z],[x+w/2,y+.015,z]],'#D7E4E7',.55))
    # Masonry bands stay between window rows, preserving visible facade structure.
    for z in [.12,.22,1.04,1.14,2.0,2.1,2.99,3.10]:
        lines.append(([[-3.2,.045,z],[3.2,.045,z]],'#C4AD87',.32))
    for xx in np.arange(-3.1,3.2,.42):
        lines.append(([[xx,-1.55,3.32],[xx,-.04,3.32]],'#A58160',.35))
    # Side wall windows and wing windows turn the architecture into a coherent 3D object.
    for y in [-1.05,-.35]:
        for z in [.62,1.58,2.54]:
            face([[3.225,y-.19,z-.30],[3.225,y+.19,z-.30],[3.225,y+.19,z+.30],[3.225,y-.19,z+.30]],'#718D9B','#A18563')
    for y in [.55,1.55,2.55]:
        for z in [.67,1.67]:
            face([[-1.575,y-.29,z-.32],[-1.575,y+.29,z-.32],[-1.575,y+.29,z+.32],[-1.575,y-.29,z+.32]],'#658A94','#997E5C')
    for z in [.67,1.67]:
        face([[-2.85,3.225,z-.32],[-1.95,3.225,z-.32],[-1.95,3.225,z+.32],[-2.85,3.225,z+.32]],'#6E8997')
    # A low railing and two courtyard planters provide real perspective cues.
    for x in np.linspace(-.9,3.1,9):lines.append(([[x,3.9,0],[x,3.9,.55]],'#817961',.65))
    lines.append(([[-.9,3.9,.55],[3.1,3.9,.55]],'#817961',.85))
    for cx,cy in [(2.5,1.7),(-.5,2.2)]:
        face([[cx-.38,cy-.28,0],[cx+.38,cy-.28,0],[cx+.38,cy+.28,0],[cx-.38,cy+.28,0]],'#A48F6D')
        face([[cx-.36,cy-.26,.18],[cx+.36,cy-.26,.18],[cx+.36,cy+.26,.18],[cx-.36,cy+.26,.18]],'#6E8861')
    return faces,lines


FACES,LINES=geometry()
ALL=np.concatenate([v for v,_,_ in FACES])
FEATURES=np.array([[3.2,.025,3.3],[3.2,-1.6,3.3],[-1.575,3.2,2.65],
                   [-1.575,3.2,.02],[2.5,1.7,.18],[-.5,2.2,.18],
                   [-.9,3.9,.55],[3.1,3.9,.55],[3.2,.025,.02],[.7,.05,2.54]])


def image_scene(ax,rect,view,label=None,keypoints=False,layer=0):
    """Project the same mesh into a framed thumbnail; return feature coordinates."""
    existing=set(ax.get_children())
    x,y,w,h=rect
    box(ax,x,y,w,h,face='#EEF3F4',edge='#BDC9CC',r=.65,lw=.75)
    # A fixed geometric framing per camera keeps all scene content in the image.
    uv,_=project(ALL,view);lo=uv.min(0);hi=uv.max(0)
    span=hi-lo;scale=min((w-.7)/span[0],(h-.7)/span[1])
    center=(lo+hi)/2
    def convert(p):
        q,_=project(p,view);return(q-center)*scale+np.array([x+w/2,y+h/2])
    order=[]
    for v,c,e in FACES:
        _,d=project(v,view);order.append((d.mean(),v,c,e))
    for _,v,c,e in sorted(order,key=lambda item:item[0],reverse=True):
        ax.add_patch(Polygon(convert(v),closed=True,facecolor=c,edgecolor=e,lw=.35,zorder=3))
    # Retain only front-facing architectural detail lines to avoid hidden wireframes.
    for v,c,lw in LINES:
        q=convert(v);ax.plot(q[:,0],q[:,1],color=c,lw=lw*.7,zorder=4)
    pts=convert(FEATURES)
    if keypoints:
        ax.scatter(pts[:,0],pts[:,1],s=17,facecolors='none',edgecolors=BLUE,linewidths=.95,zorder=6)
    if label:
        box(ax,x+.55,y+h-2.7,3.25,2.1,face='white',edge='white',r=.3,lw=0)
        text(ax,x+2.18,y+h-1.66,label,7.5,BLUE,'bold',ha='center',zorder=7)
    for artist in ax.get_children():
        if artist not in existing:artist.set_zorder(artist.get_zorder()+layer)
    return pts


def cloud_points():
    """Sample synthetic facade/roof/ground coordinates; no measured data."""
    rng=np.random.default_rng(SEED)
    vertices=[];colors=[]
    for v,c,e in FACES:
        # Facade samples produce building-shaped sparse structure, not a solid mesh.
        count=30 if len(v)==4 else 8
        a=rng.random((count,1));b=rng.random((count,1))
        pts=v[0]+a*(v[1]-v[0])+b*(v[3]-v[0])
        vertices.extend(pts)
        tone=('#305D72' if c in ['#678696','#718D9B','#658A94','#6E8997']
              else '#96754A' if c in ['#D0B991','#D4BF9C','#D3B997','#F0DEB8']
              else '#8D6844' if c=='#C19A76' else '#68845D' if c=='#6E8861' else '#AA997D')
        colors.extend([tone]*count)
    return np.asarray(vertices),colors


CLOUD,CLOUD_COLORS=cloud_points()


def frustum(position,target=TARGET,depth=.85,width=.54,height=.36):
    f=target-position;f/=np.linalg.norm(f)
    r=np.cross(f,[0.,0.,1.]);r/=np.linalg.norm(r);u=np.cross(r,f)
    c=position+f*depth
    plane=np.array([c-r*width-u*height,c+r*width-u*height,c+r*width+u*height,c-r*width+u*height])
    return plane


def reconstruction(ax,rect,detail=True):
    x,y,w,h=rect
    view=np.array([12.,17.,14.])
    cameras=[np.array([5.,6.4,2.6]),np.array([2.4,7.3,2.8]),np.array([-.6,7.7,2.9]),np.array([-3.5,6.,3.1])]
    allpts=np.concatenate((CLOUD,np.asarray(cameras)))
    uv,_=project(allpts,view);lo=uv.min(0);hi=uv.max(0)
    span=hi-lo;scale=min(w/span[0],h/span[1]);center=(lo+hi)/2
    def convert(p):q,_=project(p,view);return(q-center)*scale+[x+w/2,y+h/2]
    q=convert(CLOUD)
    # All positions lie on the synthetic mesh surfaces; no random jitter of geometry.
    ax.scatter(q[:,0],q[:,1],s=3.4 if detail else 2.0,c=CLOUD_COLORS,alpha=.95,linewidths=0,zorder=2)
    # Accented structural edges orient the viewer without a full opaque rendering.
    for e in [[[-3.2,0,0],[3.2,0,0],[3.2,0,3.3],[-3.2,0,3.3],[-3.2,0,0]],
              [[-1.6,0,0],[-1.6,3.2,0],[-1.6,3.2,2.65],[-1.6,0,2.65]]]:
        z=convert(e);ax.plot(z[:,0],z[:,1],color='#85633E',lw=.65,alpha=.85,zorder=1)
    landmark=np.array([1.8,.05,1.58])
    star=convert([landmark])[0]
    cs=convert(cameras)
    if detail:
        for i in range(3):
            ax.plot([cs[i,0],star[0]],[cs[i,1],star[1]],color=[BLUE,TEAL,VIOLET][i],lw=1.45,alpha=.9,zorder=3)
        ax.scatter(star[0],star[1],s=135,marker='*',c='#D18428',edgecolors='white',linewidths=.6,zorder=8)
        text(ax,x+w*.995,y+h*.9,'Shared 3D\nlandmark',10.3,INK,'bold',ha='right',zorder=9)
        arrow(ax,(x+w*.78,y+h*.80),(star[0]+.6,star[1]+.4),c=MUTED,lw=.7)
    ax.plot(cs[:,0],cs[:,1],color=BLUE,lw=1,ls=(0,(2,3)),alpha=.6,zorder=3)
    for i,cam in enumerate(cameras):
        plane=frustum(cam,target=landmark);p=convert(plane);c=convert([cam])[0]
        cc=[BLUE,TEAL,VIOLET,BLUE][i]
        for vertex in p:ax.plot([c[0],vertex[0]],[c[1],vertex[1]],color=cc,lw=.7,zorder=4)
        loop=np.vstack((p,p[0]));ax.plot(loop[:,0],loop[:,1],color=cc,lw=1,zorder=4)
        ax.scatter(c[0],c[1],s=9,c=cc,zorder=5)
        if detail:text(ax,c[0],c[1]-1.8,f'C{i+1}',7.5,cc,'bold',ha='center')
    return cs


def graph(ax,center,scale=1):
    coords=np.array([[-6,-1],[-3,3],[0,-2],[3,1],[8,2],[11,-1],[9,-4]],float)*scale+center
    for a,b in [(0,1),(1,2),(0,2),(1,3),(2,3),(4,5),(4,6),(5,6)]:
        ax.plot(coords[[a,b],0],coords[[a,b],1],color=TEAL,lw=1.3,zorder=2)
    a,b=3,4
    ax.plot(coords[[a,b],0],coords[[a,b],1],color=ORANGE,lw=1,ls='--',zorder=2)
    middle=(coords[a]+coords[b])/2
    text(ax,*middle,'×',15,ORANGE,'bold',ha='center',zorder=6)
    for i,p in enumerate(coords):
        color=BLUE if i<4 else VIOLET
        ax.scatter(*p,s=53,facecolors='white',edgecolors=color,linewidths=1.3,zorder=3)
        text(ax,p[0],p[1],str(i+1),6.5,color,'bold',ha='center',zorder=5)


def similarity(ax,x,y,size=12):
    # Six views from two schematic places, deliberately a labeled visual example.
    vals=np.array([[1,.88,.71,.18,.14,.2],[.88,1,.81,.16,.11,.16],[.71,.81,1,.21,.18,.22],
                   [.18,.16,.21,1,.85,.79],[.14,.11,.18,.85,1,.87],[.2,.16,.22,.79,.87,1]])
    cmap=matplotlib.colormaps['Blues'];cell=size/6
    for i in range(6):
        for j in range(6):
            ax.add_patch(Polygon([[x+j*cell,y+(5-i)*cell],[x+(j+1)*cell,y+(5-i)*cell],
                                  [x+(j+1)*cell,y+(6-i)*cell],[x+j*cell,y+(6-i)*cell]],
                                 facecolor=cmap(.06+.84*vals[i,j]),edgecolor='white',lw=.6))
    return vals


def descriptor(ax,x,y,w=17,h=2):
    values=[.14,.32,.9,.52,.15,.66,.82,.42,.24,.94,.62,.35,.73,.13,.88,.48]
    for i,v in enumerate(values):
        xx=x+i*w/len(values)
        ax.add_patch(Polygon([[xx,y],[xx+w/len(values)-.13,y],[xx+w/len(values)-.13,y+h*v],[xx,y+h*v]],
                             facecolor=matplotlib.colormaps['Blues'](.3+.65*v),edgecolor='none'))


def build():
    style={'font.family':'DejaVu Sans','svg.fonttype':'none','figure.facecolor':'white',
           'savefig.facecolor':'white','svg.hashsalt':'sfm-method-illustration-v1'}
    with plt.rc_context(style):
        fig=plt.figure(figsize=(14,12));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,120),ylim=(0,110));ax.axis('off')
        text(ax,4,106,'SFM SCENE SEPARATION STUDY',9.5,BLUE,'bold')
        text(ax,4,100.7,'From image collections to 3D scene structure',25.2,INK,'bold')
        text(ax,4,95.8,'I inspect the connections, the reconstructed geometry, and the scene boundaries together.',11.2,MUTED)
        # Overview: depict data transformations instead of another row of empty text boxes.
        image_scene(ax,(4,85.2,11.4,8.3),VIEWS[0],label='V1')
        image_scene(ax,(16.2,85.2,11.4,8.3),VIEWS[1],label='V2')
        image_scene(ax,(10.1,76.4,11.4,8.3),VIEWS[2],label='V3')
        arrow(ax,(28.3,85.4),(32.2,85.4),BLUE,lw=1.8)
        descriptor(ax,34,91.8,17,2)
        similarity(ax,35.4,77.5,12.3)
        text(ax,49.7,84.1,'→',17,BLUE,'bold',ha='center')
        text(ax,52.2,86.5,'1–2\n2–3\n4–5',9.1,BLUE,ha='center')
        arrow(ax,(55.8,85.4),(59.4,85.4),BLUE,lw=1.8)
        graph(ax,np.array([67.2,84.4]),.84)
        arrow(ax,(82,85.4),(85.4,85.4),BLUE,lw=1.8)
        reconstruction(ax,(86,77.3,29.3,17.5),detail=False)
        for x,lab,sub in [(4,'01  Multiple views','Views from an unordered collection'),(33,'02  Candidate pairs','Descriptors → shortlist / all pairs'),(61,'03  Verified graph','Illustrative mixed-scene connections'),(89,'04  Sparse reconstruction','3D points + camera poses')]:
            text(ax,x,73.7,lab,11.1,INK,'bold');text(ax,x,70.1,sub,8.2,MUTED)
        # Open, paper-style comparison lanes: small match symbols, no filled cards.
        configs=[(4,'Learned matching','ALIKED → LightGlue','DINOv2 shortlist · 1024 px · cap 4600',BLUE),
                 (43,'SIFT + shortlist','SIFT → nearest neighbour','DINOv2 shortlist · 1024 px · cap 4600',TEAL),
                 (82,'SIFT + exhaustive','SIFT → nearest neighbour','All image pairs · 3200 px · cap 8192',VIOLET)]
        for lane,(x,name,desc,cap,c) in enumerate(configs):
            # Tiny paired descriptor / keypoint glyphs suggest the front-end operation.
            left=np.array([[x+1,64.6],[x+1,62.8],[x+1,61.0]])
            right=np.array([[x+6,64.2],[x+6,62.4],[x+6,60.6]])
            for i in range(3):
                pairs=range(3) if lane==2 else [i]
                for j in pairs:
                    ax.plot([left[i,0],right[j,0]],[left[i,1],right[j,1]],
                            color=c,lw=.65,alpha=.30 if lane==2 else .65)
            if lane==0:
                for xx,yy in list(left)+list(right):
                    for k,h in enumerate([.55,1.,.72]):
                        ax.plot([xx+(k-1)*.35,xx+(k-1)*.35],[yy-h/2,yy+h/2],color=c,lw=1.6)
            else:
                ax.scatter(left[:,0],left[:,1],s=13,facecolors='white',edgecolors=c,linewidths=.9,zorder=5)
                ax.scatter(right[:,0],right[:,1],s=13,facecolors='white',edgecolors=c,linewidths=.9,zorder=5)
            text(ax,x+9,65.0,name,12.3,INK,'bold')
            text(ax,x+9,62.0,desc,10.0,MUTED)
            text(ax,x+9,59.5,cap,8.0,MUTED)
            ax.plot([x+17,x+17],[57.9,56.0],color=c,lw=.8)
        ax.plot([21,99],[56.0,56.0],color='#98AAB5',lw=.85)
        arrow(ax,(60,56.0),(60,53.5),MUTED,lw=.85)
        text(ax,60,51.7,'COLMAP geometry  →  shared incremental mapper  →  evaluation',11.3,INK,'bold',ha='center')
        text(ax,60,46.8,'Same shortlist procedure; realized pairs, feature counts and camera import paths need explicit checks.',8.4,MUTED,ha='center')
        # Lower zoom-in panels: local image evidence and reconstructed output.
        box(ax,4,7.2,54.5,36.8,'#FFFFFF',RULE,1.3)
        box(ax,61.5,7.2,54.5,36.8,'#FFFFFF',RULE,1.3)
        text(ax,6.1,41.3,'A   What connects two views?',14,INK,'bold')
        text(ax,63.6,41.3,'B   What does the mapper recover?',14,INK,'bold')
        # Correspondences between identical geometric landmarks in two camera views.
        p1=image_scene(ax,(6.1,24.1,21.9,14.9),VIEWS[0],label='V1',keypoints=True)
        p2=image_scene(ax,(34.3,24.1,21.9,14.9),VIEWS[2],label='V3',keypoints=True)
        for i in [0,1,2,4,6,8]:
            ax.plot([p1[i,0],p2[i,0]],[p1[i,1],p2[i,1]],color=[BLUE,TEAL,VIOLET][i%3],lw=.65,alpha=.65,zorder=7)
            ax.scatter([p1[i,0],p2[i,0]],[p1[i,1],p2[i,1]],s=14,c=[BLUE,TEAL,VIOLET][i%3],edgecolors='white',linewidths=.3,zorder=8)
        for a,b in [(3,7),(5,0)]:
            ax.plot([p1[a,0],p2[b,0]],[p1[a,1],p2[b,1]],color=ORANGE,lw=.85,ls='--',alpha=.8,zorder=6)
        text(ax,17.1,22.2,'Keypoints',10.6,BLUE,'bold',ha='center')
        text(ax,45.2,22.2,'Putative matches',10.6,VIOLET,'bold',ha='center')
        # Tokens and a clean mini verification legend, no invented measurements.
        descriptor(ax,7.5,16.4,17.1,2.7)
        text(ax,16,13.3,'Local descriptors',8.9,MUTED,ha='center')
        arrow(ax,(26.3,17.8),(30,17.8),TEAL)
        for a,b,c in [(32,33.6,BLUE),(34.2,35.8,TEAL),(36.4,38,VIOLET)]:
            ax.plot([a,b],[18.8,18.8],color=c,lw=2)
        text(ax,39,18.8,'verified',9.4,TEAL,'bold')
        ax.plot([32,38],[14.9,14.9],color=ORANGE,lw=1.4,ls='--');text(ax,39,14.9,'rejected',9.4,ORANGE,'bold')
        text(ax,6.2,9.4,'Solid: verified · dashed: rejected · colors: different correspondences',8.0,MUTED)
        # The same architectural scene is sampled into sparse 3D points.
        reconstruction(ax,(65,19.6,45.6,19),detail=True)
        # Camera axes at the bottom illustrate downstream checks rather than numerical results.
        text(ax,64.3,15.9,'Inspect the output',9.3,INK,'bold')
        for cx,lab,c in [(66.5,'A',BLUE),(72.5,'B',VIOLET)]:
            for dx,dy in [(-1,0),(.2,.9),(1.1,-.5)]:
                ax.scatter(cx+dx,10.9+dy,s=14,c=c,zorder=5)
            box(ax,cx-2.15,8.9,4.4,4.25,'none',c,.9,.8)
            text(ax,cx,10.9,lab,6.8,'white','bold',ha='center')
        text(ax,78.3,11.3,'Scene grouping  ·  positions  ·  orientations',8.5,MUTED)
        text(ax,4,3.3,'Illustrative geometry · not an experimental reconstruction',9.6,INK,'bold')
        text(ax,116,3.3,'Original projected 3D scene  /  no external images',8.2,MUTED,ha='right')
        OUT.mkdir(exist_ok=True)
        name='method-visual'
        fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white',transparent=False)
        desc=('Original illustrative graphical abstract, not an experimental reconstruction. Three perspective images of a deterministic L-shaped courtyard lead to schematic descriptors and a similarity matrix, a verified image graph, and sparse 3D points with camera frusta. Three colored imaging rays meet at a shared facade landmark. Three compact comparison arms join before COLMAP geometry and the shared incremental mapper. Lower panels zoom into projected landmark correspondences and the same synthetic architecture reconstructed as an illustrative point cloud. All displayed image geometry, descriptor values and graph links are explanatory examples, not measured results.')
        fig.savefig(OUT/f'{name}.svg',facecolor='white',transparent=False,metadata={'Date':None,'Creator':'tools/make_method_visual.py','Description':desc})
        p=OUT/f'{name}.svg';svg=p.read_text().replace("font-family: 'DejaVu Sans'","font-family: 'DejaVu Sans', Arial, Helvetica, sans-serif")
        i=svg.index('>',svg.index('<svg'))+1
        svg=svg[:i]+'\n<title>Illustrated SfM method</title>\n<desc>'+html.escape(desc)+'</desc>'+svg[i:]
        p.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n');ET.parse(p)
        plt.close(fig)
    record={'name':'method-visual','description':desc,'files':['method-visual.svg','method-visual.png'],
            'width_inches':14,'height_inches':12,'png_dpi':180,'background':'#FFFFFF',
            'svg_text':'editable, fonts not embedded','svg_font_fallback':['DejaVu Sans','Arial','Helvetica','sans-serif'],
            'png_font':'DejaVu Sans (rendered by Matplotlib)','sources':[],
            'transformations':['Hand-built L-shaped courtyard mesh in a shared synthetic 3D coordinate system.','Perspective projection from three explicitly defined cameras, with projected landmark correspondences.','Deterministic point sampling on the same mesh; seed 20260920. Three camera rays converge on the same synthetic facade landmark.','Illustrative descriptors, similarity matrix and graph links; no empirical quantities.'],
            'evidence_status':'schematic','uncertainty':'Not applicable; no experimental observations are plotted.',
            'geometry_seed':SEED,'geometry_method':'pinhole perspective projection of original parametric courtyard mesh; seeded uniform surface samples',
            'generator':'tools/make_method_visual.py','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    (OUT/'method-visual.manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Wrote figures/method-visual.svg, .png and .manifest.json')
    return record


if __name__=='__main__':build()
