#!/usr/bin/env python3
"""Render the study's diagrams and reported-result plots as editable SVG and PNG.

Run: python tools/make_figures.py
Requires: matplotlib (validated with 3.10.8); no model weights or network access.
The CSV inputs are transcriptions of historical summaries, not newly run experiments.
The original conflicting histogram is preserved under figures/archive/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import runpy
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'results' / 'reported'
INK = '#172D40'
MUTED = '#516474'
RULE = '#D6DFE6'
SOFT = '#F3F6F9'
COLORS = ['#176F9F', '#25796B', '#7053A1']
PALE = ['#EAF3F9', '#EAF4F0', '#F1EEF8']
CONFIGS = ['learned', 'colmap-shortlist', 'colmap-default']
MARKERS = ['o', 's', '^']
DPI = 180
RECORD = 'Recorded summaries • original experiments not rerun in this repository refresh'
MANIFEST = []


def rows(name):
    with (DATA / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, name, desc, sources, transforms):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=DPI, facecolor='white', transparent=False)
    fig.savefig(OUT / f'{name}.svg', facecolor='white', transparent=False,
                metadata={'Date': None, 'Creator': 'tools/make_figures.py', 'Description': desc})
    # SVG title/description remain native, searchable text for assistive technology.
    path = OUT / f'{name}.svg'
    svg = path.read_text()
    # Keep native text while avoiding serif fallback on systems without DejaVu.
    svg = svg.replace("font-family: 'DejaVu Sans'",
                      "font-family: 'DejaVu Sans', Arial, Helvetica, sans-serif")
    import html
    marker = svg.index('>', svg.index('<svg')) + 1
    svg = svg[:marker] + '\n<title>' + html.escape(name.replace('-', ' ')) + '</title>\n<desc>' + html.escape(desc) + '</desc>' + svg[marker:]
    path.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n')
    ET.parse(path)
    MANIFEST.append({'name': name, 'description': desc, 'files': [f'{name}.svg', f'{name}.png'],
                     'width_inches': fig.get_figwidth(), 'height_inches': fig.get_figheight(),
                     'png_dpi': DPI, 'background': '#FFFFFF', 'svg_text': 'editable, fonts not embedded',
                     'svg_font_fallback': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
                     'png_font': 'DejaVu Sans (rendered by Matplotlib)',
                     'sources': sources, 'transformations': transforms, 'evidence_status': 'schematic' if not sources else 'historical_summary_not_rerun',
                     'uncertainty': 'No replicate-level uncertainty available; no intervals inferred.'})
    plt.close(fig)
    print(f'Wrote {name}.svg and .png')


def canvas(height=9):
    fig = plt.figure(figsize=(13, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis('off')
    return fig, ax


def txt(ax, x, y, text, size=12, weight='normal', color=INK, ha='left', va='center', **kw):
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
                   linespacing=1.45, **kw)


def box(ax, x, y, w, h, face=SOFT, edge=RULE, radius=1.3, lw=1.1):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                          linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, lw=1.8, style='-|>', connection='arc3', **kw):
    p = FancyArrowPatch(start, end, arrowstyle=style, color=color, linewidth=lw,
                        mutation_scale=14, connectionstyle=connection, **kw)
    ax.add_patch(p)


def header(ax, kicker, title, subtitle):
    txt(ax, 3.5, 96, kicker.upper(), 10, 'bold', COLORS[0])
    txt(ax, 3.5, 91, title, 24, 'bold')
    txt(ax, 3.5, 86, subtitle, 11.8, color=MUTED)


def framework():
    fig, ax = canvas(9.4)
    header(ax, 'Study design', 'When do image collections become the wrong 3D scenes?',
           'I compare three front ends, then inspect separation, camera geometry and reconstruction cost.')
    for x, label in [(3.5, '01  INPUT'), (22, '02  PAIR SELECTION'), (44, '03  LOCAL FEATURES'), (72.5, '04  SHARED BACK END')]:
        txt(ax, x, 79.5, label, 10, 'bold', MUTED)
    box(ax, 3.5, 38, 15, 37)
    # Three abstract image cards, explicitly an icon rather than source imagery.
    for dx, dy in [(0,0),(1.2,1.2),(2.4,2.4)]:
        box(ax, 7+dx, 63+dy, 7, 5.2, 'white', '#9DAFBD', .5)
        ax.plot([7.6+dx,9+dx,10.1+dx,12.7+dx],[64+dy,66.2+dy,64.9+dy,67+dy],color='#9DAFBD',lw=1)
    txt(ax, 11, 57.5, 'Unordered\nimage collection', 12.4, 'bold', ha='center')
    txt(ax, 11, 45.7, 'Single / mixed\nRevisit / UAV', 11.3, color=MUTED, ha='center')

    box(ax, 22, 51, 18, 24, '#EDF3F7', '#ABBCCA')
    txt(ax,31,69,'DINOv2 shortlist',13,'bold',ha='center')
    txt(ax,31,61,'Global descriptors\nL2 distance + neighbour floor',10.3,ha='center',color=MUTED)
    txt(ax,31,53.7,'Same shortlist procedure',9.8,'bold',ha='center')
    box(ax,22,32,18,13,PALE[2],COLORS[2])
    txt(ax,31,40.2,'Exhaustive pairs',12.5,'bold',ha='center')
    txt(ax,31,35.4,'All image combinations',10.4,ha='center',color=MUTED)
    arrow(ax,(18.5,64),(22,64)); arrow(ax,(18.5,42),(22,39))

    for i,(y,head,detail) in enumerate([
        (64,'ALIKED + LightGlue','learned · 1024 px · cap 4600'),
        (48,'SIFT + nearest neighbour','colmap-shortlist · 1024 px · cap 4600'),
        (32,'SIFT + nearest neighbour','colmap-default · 3200 px · cap 8192')]):
        box(ax,44,y,25,11.5,PALE[i],COLORS[i])
        txt(ax,45.5,y+8,head,12,'bold',COLORS[i])
        txt(ax,45.5,y+3.5,detail,9.5,color=MUTED)
        arrow(ax,(40,69.6 if i==0 else 53.7 if i==1 else 38.5),(44,y+5.8),COLORS[i])
    txt(ax,45,28.1,'Detector and matcher change together.',10.5,'bold')
    txt(ax,45,24.4,'Feature counts and camera initialization/import paths differ.',9.8,color=MUTED)

    # Verification is implemented through arm-specific database paths; mapping/evaluation are shared.
    box(ax,72.5,32,24,43,'#F2F5F8','#ABBCCA')
    txt(ax,84.5,69.7,'COLMAP geometry',12.7,'bold',ha='center')
    txt(ax,84.5,64.5,'Verified image-pair graph',10.7,ha='center',color=MUTED)
    arrow(ax,(84.5,60.6),(84.5,57.4))
    txt(ax,84.5,54.3,'Incremental mapping',12.7,'bold',ha='center')
    txt(ax,84.5,49.2,'Same CPU call and options',10.7,ha='center',color=MUTED)
    arrow(ax,(84.5,45.4),(84.5,42.7))
    txt(ax,84.5,39.4,'Evaluate every reconstruction',11.5,'bold',ha='center')
    for i,y in enumerate([69.8,53.8,37.8]):
        ax.plot([69,70.6],[y,y],color=COLORS[i],lw=1.8)
    ax.plot([70.6,70.6],[37.8,69.8],color=MUTED,lw=1.6)
    arrow(ax,(70.6,69.8),(72.5,69.8),MUTED)

    for x,w,title,detail in [(3.5,28,'SEPARATION','Registered images · cluster count\nPurity · verified-pair precision'),
                              (35.5,29,'GEOMETRY','Relative-pose AUC · Sim(3) alignment\nCamera positions AND orientations'),
                              (68.5,28,'COST & CONTROLS','CPU mapping time\nDevice, resolution and feature budget')]:
        box(ax,x,8.5,w,12.5,'white',RULE)
        txt(ax,x+1.4,17.6,title,10.3,'bold',COLORS[0])
        txt(ax,x+1.4,12.4,detail,10.4,color=MUTED)
    ax.plot([84.5,84.5,17.5],[32,22.8,22.8],color=MUTED,lw=1.6)
    for x in [17.5,50,84.5]: arrow(ax,(x,22.8),(x,21),lw=1.6)
    txt(ax,3.5,3.4,'Ongoing study • framework schematic, not a reconstruction • historical results require fresh runs and archived outputs',10,color=MUTED)
    save(fig,'framework-overview','Three arms compare DINOv2 retrieval with ALIKED and LightGlue, the same retrieval procedure with SIFT and nearest-neighbour matching, and exhaustive SIFT matching. All lead to COLMAP mapping and evaluation of separation, pose and CPU cost. Feature counts and camera initialization/import paths differ; this is not a matcher-only ablation.',[],['Hand-drawn schematic from src/sfm_pipeline.py, src/baseline_colmap.py and src/run_experiments.py; no empirical coordinates.'])


def graph(ax, center, labels, colors, edges, scale=1, dashed=()):
    coords = [(-3,1.8),(-1,4),(2.1,3),(3.7,.2),(.1,-2.1),(-3.3,-1.6)]
    coords=[(center[0]+x*scale,center[1]+y*scale) for x,y in coords[:len(labels)]]
    for a,b in edges:
        ax.plot([coords[a][0],coords[b][0]],[coords[a][1],coords[b][1]],color=MUTED,lw=1.4,zorder=1)
    for a,b in dashed:
        ax.plot([coords[a][0],coords[b][0]],[coords[a][1],coords[b][1]],color='#B95135',lw=2,ls='--',zorder=2)
    for (x,y),label,c in zip(coords,labels,colors):
        ax.scatter(x,y,s=200,facecolors='white',edgecolors=c,linewidths=2,zorder=3)
        txt(ax,x,y,label,9,'bold',c,ha='center',zorder=4)
    return coords


def camera(ax,x,y,direction,color):
    # Drawing lives in the same schematic 2D coordinate system as the diagram.
    dx,dy=math.cos(direction),math.sin(direction)
    nx,ny=-dy,dx
    points=[(x,y),(x+dx*4+nx*1.9,y+dy*4+ny*1.9),(x+dx*4-nx*1.9,y+dy*4-ny*1.9)]
    ax.add_patch(Polygon(points,closed=True,facecolor='none',edgecolor=color,lw=1.8))
    ax.add_patch(Circle((x,y),.6,facecolor=color,edgecolor='white',lw=.8,zorder=4))


def failures():
    fig,ax=canvas(8.4)
    header(ax,'What I check','Three ways a plausible model can still be wrong',
           'Conceptual examples only. Nodes, links and camera locations below are drawn illustrations.')
    for x,title,sub in [(3.5,'A  FALSE MERGE','Different places become one model'),(35.3,'B  FALSE SPLIT','The same place becomes two models'),(67.1,'C  ORIENTATION ERROR','Camera centres hide a pose failure')]:
        box(ax,x,21,29.4,57,'white',RULE)
        txt(ax,x+1.5,73.5,title,12.3,'bold')
        txt(ax,x+1.5,68,sub,10.5,color=MUTED)
    # Input identities then a graph with a spurious bridge.
    txt(ax,18.2,60.5,'Place A       Place B',11,'bold',ha='center')
    pa=graph(ax,(12,46),['A']*3,[COLORS[0]]*3,[(0,1),(1,2),(0,2)],1.05)
    pb=graph(ax,(24,43),['B']*3,[COLORS[2]]*3,[(0,1),(1,2),(0,2)],1.05)
    arrow(ax,pa[2],pb[0],'#B95135',style='-',linestyle='--',lw=2)
    box(ax,6,37,24,17,face='none',edge='#B95135',lw=1.4)
    txt(ax,18.2,32.5,'Spurious verified connection',10.5,'bold','#B95135',ha='center')
    txt(ax,18.2,25.5,'Need: separate scene components',10.2,ha='center',color=MUTED)
    # Disconnected session groups of the same identity.
    txt(ax,50,60.5,'Same place, two sessions',11,'bold',ha='center')
    graph(ax,(43,46),['A']*3,[COLORS[1]]*3,[(0,1),(1,2),(0,2)],1.0)
    graph(ax,(56.5,43),['A']*3,[COLORS[1]]*3,[(0,1),(1,2),(0,2)],1.0)
    box(ax,37.5,40,11.2,15,face='none',edge=COLORS[1]);box(ax,51,37,11.2,15,face='none',edge=COLORS[1])
    txt(ax,50,32.5,'Missing or insufficient bridge',10.5,'bold','#B95135',ha='center')
    txt(ax,50,25.5,'Need: one connected reconstruction',10.2,ha='center',color=MUTED)
    # Shared camera positions and disagreeing optical axes.
    txt(ax,81.8,60.5,'One possible error: reversed view',10.5,'bold',ha='center')
    for x,y in [(74,46),(82,43),(90,48)]:
        camera(ax,x,y,math.pi/2,COLORS[1]);camera(ax,x,y,-math.pi/2,'#B95135')
    txt(ax,81.8,36,'Expected ↑       Estimated ↓',10.5,ha='center')
    txt(ax,81.8,30.5,'Inspect camera orientations',10.5,'bold','#B95135',ha='center')
    txt(ax,81.8,25.5,'Check full 3D rotation, including roll',10.2,ha='center',color=MUTED)
    box(ax,3.5,7,93,9,SOFT,RULE)
    txt(ax,6,11.5,'Registration and cluster count are necessary checks; ground-truth pose evaluation can expose errors they miss.',11.5)
    save(fig,'failure-modes','Three explicitly schematic examples: a false cross-scene bridge merges different places, insufficient cross-session connectivity splits a single place, and matching camera centres conceal incorrect orientations. No plotted points are reconstructed data.',[],['Illustrative graph layouts and camera frusta are manually positioned, with no experimental data.'])


def polish(ax, ylabel=None):
    ax.set_facecolor('white')
    ax.spines[['top','right']].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED,labelsize=10,length=3)
    ax.grid(axis='y',color=RULE,lw=.7,zorder=0)
    ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel,color=MUTED,fontsize=10.5)


def chart_figure(title,subtitle,height=9.3):
    fig, axs=plt.subplots(2,2,figsize=(13,height))
    fig.subplots_adjust(left=.077,right=.97,bottom=.145,top=.785,hspace=.53,wspace=.25)
    fig.text(.045,.962,title,fontsize=23,fontweight='bold',color=INK,va='top')
    fig.text(.045,.908,subtitle,fontsize=11.6,color=MUTED,va='top',linespacing=1.45)
    handles=[Line2D([0],[0],marker=MARKERS[i],color=COLORS[i],label=c,linewidth=0,markersize=8) for i,c in enumerate(CONFIGS)]
    fig.legend(handles=handles,ncol=3,loc='upper left',bbox_to_anchor=(.039,.869),frameon=False,fontsize=11)
    fig.text(.045,.027,RECORD,fontsize=9.5,color=MUTED)
    return fig,axs


def panel_title(ax,text):
    ax.set_title(text,loc='left',fontweight='bold',fontsize=12.4,pad=13,color=INK)


def grouped(ax,data,key,groups,kind='point',fmt='.3f',ylim=None,log=False,labels=None):
    xs=np.arange(len(groups)); width=.23
    for i,c in enumerate(CONFIGS):
        values=[]
        for g in groups:
            found=[r for r in data if r.get('scene',r.get('run'))==g and r['config']==c]
            values.append(float(found[0][key]) if found else np.nan)
        pos=xs+(i-1)*width
        if kind=='bar':
            ax.bar(pos,values,width=.20,color=COLORS[i],edgecolor='white',linewidth=.6,zorder=3)
        else:
            ax.scatter(pos,values,s=49,color=COLORS[i],marker=MARKERS[i],zorder=4)
        for j,(x,v) in enumerate(zip(pos,values)):
            if math.isfinite(v):
                label=format(v,fmt) if labels is None else labels[i][j]
                ax.annotate(label,(x,v),xytext=(0,21 if i==1 else 8),textcoords='offset points',ha='center',fontsize=9.5,color=COLORS[i],fontweight='bold')
            else:
                ax.text(x, .04, 'NR',transform=ax.get_xaxis_transform(),ha='center',fontsize=9,color=MUTED)
    ax.set_xticks(xs,groups);ax.set_xlim(-.6,len(groups)-.4)
    if log: ax.set_yscale('log',base=10)
    if ylim: ax.set_ylim(*ylim)
    polish(ax)


def single_results():
    data=rows('single-scenes.csv'); groups=['pipes','terrace','courtyard']
    fig,axs=chart_figure('Single-scene reconstruction is configuration dependent',
        'ETH3D • 14 pipes, 23 terrace and 38 courtyard images • all values below are recorded summaries')
    ax=axs[0,0];panel_title(ax,'A   Relative pose AUC@5°  ↑')
    grouped(ax,data,'auc5',groups,ylim=(0,1.12));ax.set_ylabel('Normalized AUC (0–1)',fontsize=10.5,color=MUTED)
    ax=axs[0,1];panel_title(ax,'B   Median camera-centre error  ↓')
    grouped(ax,data,'median_position_mm',groups,fmt='.0f',log=True,ylim=(2,3000));ax.set_ylabel('Position error (mm; log₁₀ scale)',fontsize=10.5,color=MUTED)
    ax=axs[1,0];panel_title(ax,'C   Registered images  ↑')
    q=[dict(r,registered_pct=100*int(r['registered'])/int(r['total'])) for r in data]
    lab=[[f"{next(r for r in data if r['scene']==g and r['config']==c)['registered']}/{next(r for r in data if r['scene']==g and r['config']==c)['total']}" for g in groups] for c in CONFIGS]
    grouped(ax,q,'registered_pct',groups,kind='bar',ylim=(0,118),labels=lab);ax.set_ylabel('Registered / total (%)',fontsize=10.5,color=MUTED);ax.set_yticks([0,25,50,75,100])
    ax=axs[1,1];panel_title(ax,'D   CPU incremental mapping  ↓')
    grouped(ax,data,'mapping_s',groups,kind='bar',fmt='.1f',ylim=(0,185));ax.set_ylabel('Mapping time (s)',fontsize=10.5,color=MUTED)
    fig.text(.045,.084,'Controls: shortlist arms use the same retrieval procedure and 1024 px input; realized pairs require saved outputs.\nFeature counts and camera initialization/import paths differ. These comparisons do not isolate matcher causality.',fontsize=10.4,color=MUTED,linespacing=1.45)
    save(fig,'results-single-scenes','Four panels show recorded single-scene AUC at 5 degrees, median camera-centre error in millimetres on a base-10 log scale, registered image counts, and CPU mapping time. SIFT shortlist registers 8/14 pipes images; learned courtyard has 1111 mm median position error. Comparisons change detector and matcher together, and default also changes pairing and budget.', ['results/reported/single-scenes.csv'],['Registration plotted as 100 × registered / total, with original counts annotated.','Position uses a base-10 logarithmic axis; no zeros or missing observations.','AUC5 shown; AUC10, AUC20 and rotation remain in source table.','Bars start at zero. No error bars or aggregation across scenes.'])


def mixed_results():
    mix=rows('mixed-scenes.csv');rev=rows('revisit.csv');poses=rows('mixed-scene-poses.csv')
    data=[dict(r,run='Mixed scenes') for r in mix]+[dict(r,run='Revisit') for r in rev]
    fig,axs=chart_figure('Scene separation must work in both directions',
        'Mixed scenes: three distinct places, 75 images   |   Revisit: one place in two sessions, 62 images')
    ax=axs[0,0];panel_title(ax,'A   Reconstructed scene count')
    grouped(ax,data,'clusters',['Mixed scenes','Revisit'],kind='bar',fmt='.0f',ylim=(0,4.05))
    for x,y in [(0,3),(1,1)]:
        ax.plot([x-.45,x+.45],[y,y],color=INK,ls='--',lw=1.2,zorder=5)
        ax.annotate(f'expected {y}',(x+.28 if x==1 else x-.32,y),xytext=(0,22),textcoords='offset points',ha='center',fontsize=9.5,color=MUTED)
    ax.set_ylabel('Reconstructions (count)',fontsize=10.5,color=MUTED);ax.set_yticks([0,1,2,3,4])
    ax=axs[0,1];panel_title(ax,'B   Mixed-scene verification and purity  ↑')
    quality=[]
    for r in mix:
        for k,l in [('pair_precision','Pair precision'),('purity','Cluster purity')]:quality.append(dict(r,scene=l,value=r[k]))
    grouped(ax,quality,'value',['Pair precision','Cluster purity'],fmt='.4f',ylim=(0,1.17))
    ax.set_ylabel('Fraction (0–1)',fontsize=10.5,color=MUTED)
    ax=axs[1,0];panel_title(ax,'C   Pose error inside the mixed collection  ↓')
    grouped(ax,poses,'median_position_mm',['pipes','terrace','courtyard'],fmt='.0f',log=True,ylim=(2,7000));ax.set_ylabel('Position error (mm; log₁₀ scale)',fontsize=10.5,color=MUTED)
    ax=axs[1,1];panel_title(ax,'D   CPU incremental mapping  ↓')
    grouped(ax,data,'mapping_s',['Mixed scenes','Revisit'],kind='bar',fmt='.1f',ylim=(0,660));ax.set_ylabel('Mapping time (s)',fontsize=10.5,color=MUTED)
    fig.text(.045,.085,'Mixed: learned 75/75 registered, shortlist 69/75, default 75/75; learned terrace includes 8/23 cameras with >170° rotation error.\nRevisit: both reported arms register 62/62; cross-session links are true links. NR = no default-arm result reported.',fontsize=10.4,color=MUTED,linespacing=1.45)
    save(fig,'results-mixed-scenes','Four panels compare recorded mixed-scene and revisit cluster counts, mixed-scene verified-pair precision and cluster purity, mixed-scene median camera-centre errors, and CPU mapping time. Expected clusters are three for mixed and one for revisit. The learned arm reports two in both. The default revisit configuration was not reported, shown as NR rather than zero.', ['results/reported/mixed-scenes.csv','results/reported/mixed-scene-poses.csv','results/reported/revisit.csv'],['Plot recorded cluster counts and metric fractions directly.','Position uses a base-10 logarithmic axis.','Missing colmap-default revisit result remains missing (NR).','Expected counts are reference lines, not observations.','No uncertainty inferred.'])


def uav_results():
    data=rows('uav.csv')
    fig,axs=chart_figure('Complete registration can conceal large pose errors',
        'Mill 19 building • 120 consecutive frames • all three arms report 120/120 registered in one reconstruction')
    ax=axs[0,0];panel_title(ax,'A   Relative pose accuracy  ↑')
    for i,r in enumerate(data):
        vals=[float(r[k]) for k in ['auc5','auc10','auc20']]
        ax.plot([5,10,20],vals,color=COLORS[i],marker=MARKERS[i],lw=1.7,markersize=7)
        for x,y in zip([5,10,20],vals):
            offset=12 if i==2 else -18 if i==1 else 10
            ax.annotate(f'{y:.3f}',(x,y),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9.5,color=COLORS[i],fontweight='bold')
    ax.set(xlim=(2.5,22.5),ylim=(0,1.12),xticks=[5,10,20],xlabel='Angular threshold (degrees)',ylabel='Normalized AUC (0–1)');polish(ax)
    def one(ax,key,log=False,fmt='.2f',yl=(0,1),ylabel=''):
        for i,r in enumerate(data):
            v=float(r[key]);ax.scatter(i,v,s=65,color=COLORS[i],marker=MARKERS[i],zorder=4)
            ax.annotate(format(v,fmt),(i,v),xytext=(0,10),textcoords='offset points',ha='center',color=COLORS[i],fontsize=11,fontweight='bold')
        ax.set_xticks([0,1,2],['learned','shortlist','default']);ax.set_xlim(-.6,2.6)
        if log: ax.set_yscale('log',base=10)
        ax.set_ylim(*yl);polish(ax,ylabel)
    ax=axs[0,1];panel_title(ax,'B   Median camera-centre error  ↓')
    one(ax,'median_position_u',True,'.3f',(.0004,.3),'Position error (u; log₁₀ scale)')
    ax=axs[1,0];panel_title(ax,'C   Median absolute rotation error  ↓')
    one(ax,'median_rotation_deg',False,'.2f',(0,84),'Rotation error (degrees)')
    ax.text(.40,.82,'27/120 learned cameras\n>170° rotation error\n0/120 in each SIFT arm',transform=ax.transAxes,fontsize=10.3,color=MUTED,va='top',linespacing=1.4)
    ax=axs[1,1];panel_title(ax,'D   CPU incremental mapping  ↓')
    vals=[float(r['mapping_s']) for r in data]
    ax.bar([0,1,2],vals,color=COLORS,width=.55,zorder=3)
    for i,v in enumerate(vals):ax.annotate(f'{v:.1f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',color=COLORS[i],fontsize=11,fontweight='bold')
    ax.set_xticks([0,1,2],['learned','shortlist','default']);ax.set_ylim(0,2750);polish(ax,'Mapping time (s)')
    fig.text(.045,.085,'u = Mega-NeRF normalized coordinate unit, not metres; it cannot be compared with ETH3D millimetres.\nThe shortlist SIFT arm reports 7592 keypoints/image versus ALIKED’s 4600. Front-end CPU/GPU timings are omitted.',fontsize=10.4,color=MUTED,linespacing=1.45)
    save(fig,'results-uav','Four panels show recorded Mill 19 AUC at 5, 10 and 20 degrees, median position error in normalized units on a log scale, median rotation error, and CPU mapping time. All configurations register 120 images in one model, but learned has median rotation error 64.33 degrees and 27 cameras over 170 degrees. Normalized units are not metres and feature counts differ.', ['results/reported/uav.csv'],['Plot AUC at its three reported angular thresholds, connecting points only to identify configurations.','Position axis uses base-10 logarithm; no zeros.','Rotation axis is linear, starting at zero.','Only shared CPU mapping time is shown, not heterogeneous front-end timings.'])


def link_ranges():
    data=rows('verified-link-ranges.csv')
    fig,ax=plt.subplots(figsize=(13,5.7))
    fig.subplots_adjust(left=.24,right=.95,bottom=.29,top=.71)
    fig.text(.045,.935,'Verified links: strength alone does not establish scene identity',fontsize=20,fontweight='bold',color=INK)
    fig.text(.045,.859,'Mixed-scene learned arm • 889 within-scene and 150 cross-scene verified pairs • summary statistics only',fontsize=11.5,color=MUTED)
    fig.text(.045,.79,'Thin line: minimum–maximum     Thick line: 10th–90th percentile     Marker: median',fontsize=10.8,color=MUTED)
    for i,r in enumerate(data):
        y=1-i;c=COLORS[0] if i==0 else '#B95135'
        vals=[float(r[k]) for k in ['min','p10','median','p90','max']]
        ax.plot([vals[0],vals[4]],[y,y],c=c,lw=1.5)
        ax.plot([vals[1],vals[3]],[y,y],c=c,lw=9,solid_capstyle='butt')
        ax.scatter(vals[2],y,marker=['o','s'][i],s=85,c='white',edgecolors=c,linewidths=2,zorder=5)
        for x in [vals[0],vals[4]]:ax.plot([x,x],[y-.06,y+.06],c=c,lw=1.5)
        ax.annotate(f"median {int(vals[2])}",(vals[2],y),xytext=(0,18),textcoords='offset points',ha='center',color=c,fontsize=11,fontweight='bold')
        ax.annotate(f"max {int(vals[4])}",(vals[4],y),xytext=(0,-23),textcoords='offset points',ha='center',color=MUTED,fontsize=10)
    ax.set_xscale('log',base=10);ax.set_xlim(10,5000);ax.set_ylim(-.5,1.6)
    ax.set_yticks([1,0],['Within-scene\nTrue links (n=889)','Cross-scene\nFalse links (n=150)'])
    ax.set_xticks([10,20,50,100,200,500,1000,2000,5000],['10','20','50','100','200','500','1000','2000','5000'])
    ax.set_xlabel('RANSAC inliers per verified pair (log₁₀ scale)',fontsize=11,color=MUTED,labelpad=13)
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(RULE)
    ax.tick_params(axis='y',length=0,pad=15,labelsize=11,colors=INK);ax.tick_params(axis='x',colors=MUTED,labelsize=10)
    ax.grid(axis='x',color=RULE,lw=.7);ax.set_axisbelow(True)
    fig.text(.045,.092,'These are distribution summaries, not confidence intervals. No histogram or universal inlier cutoff is inferred.\nThe revisit experiment separately reports true cross-session links with median 51; population ranges are unavailable here.',fontsize=10.5,color=MUTED,linespacing=1.45)
    fig.text(.045,.027,RECORD,fontsize=9.5,color=MUTED)
    save(fig,'link-ranges','Recorded mixed-scene verified-link distributions shown as minimum-to-maximum lines, 10th-to-90th-percentile thick lines and median markers. Within-scene n889 ranges 15 to 3376 with median 308; false cross-scene n150 ranges 15 to 85 with median 22. These are summaries, not intervals of statistical uncertainty or evidence for a universal cutoff.', ['results/reported/verified-link-ranges.csv','results/reported/revisit-links.csv'],['Use base-10 logarithmic inlier-count axis.','Do not synthesize observations, histogram bins, unseen percentiles or uncertainty intervals.','Only RESULTS snapshot values are used; legacy conflicting histogram is archived separately.'])


def main():
    MANIFEST.clear()
    style={'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':MUTED,
           'axes.titlecolor':INK,'figure.facecolor':'white','savefig.facecolor':'white',
           'svg.fonttype':'none','svg.hashsalt':'sfm-scene-separation-study-figures-v1','pdf.fonttype':42}
    with plt.rc_context(style):
        framework();failures();single_results();mixed_results();uav_results();link_ranges()
    # The illustrated plates share this rebuild command and provenance manifest.
    method = runpy.run_path(str(ROOT / 'tools/make_method_visual.py'))
    MANIFEST.append(method['build']())
    failure = runpy.run_path(str(ROOT / 'tools/make_failure_visual.py'))
    failure['draw']()
    MANIFEST.append(json.loads((OUT / 'failure-visual.manifest.json').read_text()))
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv'))}
    manifest={'generator':'tools/make_figures.py','python':platform.python_version(),'matplotlib':matplotlib.__version__,
              'numpy':np.__version__,'source_snapshot':'RESULTS.md at commit 7e0d60c',
              'experiments_rerun':False,'source_sha256':sources,'figures':MANIFEST}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
