"""Robustness check on the inter-sortie vertical offset.

Tests whether the metre-scale offsets are an artifact of the ground estimator or the
cell filter, and whether the offset is a constant shift or a tilt. Uses the three pairs
with the most comparable cells.
"""
import sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from test_relative_accuracy import load_index, fetch_xyz, M_PER_FTUS

PAIRS=[("DRCOG_X252_Y025","I25N_X015_Y001"),
       ("DRCOG_X012_Y003","I25S2_X139_Y093"),
       ("I15South_X620_Y268","SLC_X044_Y010")]
d=load_index(glob.glob("/workspace/data/lidar/v1/stac/index/*.parquet")[0]).set_index("id")

for A,B in PAIRS:
    ra,rb=d.loc[A],d.loc[B]
    x0,x1=max(ra.minx,rb.minx),min(ra.maxx,rb.maxx)
    y0,y1=max(ra.miny,rb.miny),min(ra.maxy,rb.maxy)
    xa,ya,za=fetch_xyz(ra["s3_path"]); xb,yb,zb=fetch_xyz(rb["s3_path"])
    print(f"\n=== {A}  vs  {B} ===")
    print(f"    epsg {int(ra['proj:epsg'])} / {int(rb['proj:epsg'])}   "
          f"overlap {(x1-x0)*(y1-y0)*M_PER_FTUS**2:.0f} m2")
    # raw Z range of each tile inside the overlap, no gridding at all
    ma=(xa>=x0)&(xa<=x1)&(ya>=y0)&(ya<=y1); mb=(xb>=x0)&(xb<=x1)&(yb>=y0)&(yb<=y1)
    print(f"    points in overlap: {ma.sum():,} / {mb.sum():,}")
    for lab,zz in (("A",za[ma]),("B",zb[mb])):
        q=np.percentile(zz,[1,10,50,90,99])*M_PER_FTUS
        print(f"    {lab} Z percentiles (m): "+" ".join(f"{v:.2f}" for v in q))
    print(f"    difference of medians: {(np.median(za[ma])-np.median(zb[mb]))*M_PER_FTUS:+.3f} m")
    print(f"    difference of 1st pct: {(np.percentile(za[ma],1)-np.percentile(zb[mb],1))*M_PER_FTUS:+.3f} m")
    # is the offset constant across the overlap, or tilted?
    cell=4.0/M_PER_FTUS
    nx=max(int((x1-x0)/cell),1); ny=max(int((y1-y0)/cell),1)
    def grid(x,y,z,stat):
        ix=np.floor((x-x0)/cell).astype(int); iy=np.floor((y-y0)/cell).astype(int)
        m=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny); ix,iy,z=ix[m],iy[m],z[m]
        f=ix*ny+iy; o=np.argsort(f,kind="stable"); fs,zs=f[o],z[o]
        b=np.searchsorted(fs,np.arange(nx*ny+1)); c=np.diff(b)
        g=np.full(nx*ny,np.nan)
        for k in np.nonzero(c>=40)[0]:
            g[k]=stat(zs[b[k]:b[k+1]])
        return g
    for name,stat in (("p10",lambda v:np.percentile(v,10)),
                      ("median",np.median),
                      ("p50 of lowest 25%",lambda v:np.median(np.sort(v)[:max(len(v)//4,1)]))):
        ga,gb=grid(xa,ya,za,stat),grid(xb,yb,zb,stat)
        sel=np.isfinite(ga)&np.isfinite(gb)
        if sel.sum()<20: print(f"    {name:20s}: too few cells"); continue
        dz=(ga[sel]-gb[sel])*M_PER_FTUS
        med=np.median(dz)
        print(f"    {name:20s}: median {med:+.3f} m  NMAD {1.4826*np.median(np.abs(dz-med)):.3f}  n={sel.sum()}")
        if name=="p10":
            ii=np.nonzero(sel)[0]; cx,cy=ii//ny,ii%ny
            for ax,nm in ((cx,"easting"),(cy,"northing")):
                if len(np.unique(ax))>3:
                    sl=np.polyfit(ax*4.0,dz,1)[0]
                    print(f"        tilt along {nm}: {sl*1000:+.2f} mm per m")
