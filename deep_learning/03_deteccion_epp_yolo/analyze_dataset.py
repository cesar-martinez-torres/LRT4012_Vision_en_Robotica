"""Summarize class balance, object sizes and visual annotation examples."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_dataset import ROOT, NAMES, audit


def analyze(dataset):
    report = audit(dataset)
    output = dataset/'analysis'
    output.mkdir(exist_ok=True)
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    objects, candidates = [], defaultdict(list)
    for row in manifest:
        labels = (dataset/row['label']).read_text().splitlines()
        for line in labels:
            cls, x, y, w, h = map(float, line.split())
            objects.append({'split': row['split'], 'class': NAMES[int(cls)],
                            'area_fraction': w*h, 'width_pixels': w*row['width'],
                            'height_pixels': h*row['height'],
                            'input_area_pixels':w*row['width']*h*row['height']*(640/max(row['width'],row['height']))**2})
            if row['split']=='train':
                candidates[int(cls)].append(row)
    table = pd.DataFrame(objects)
    table.to_csv(output/'objects.csv', index=False)
    summary = table.groupby(['split','class']).agg(objects=('area_fraction','size'),
        median_area_fraction=('area_fraction','median'),
        fraction_under_32px=('input_area_pixels',lambda values: float(np.mean(values < 32*32))))
    summary.to_csv(output/'class_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    display_names=['Persona','Solo casco','Solo chaleco','Casco y chaleco']
    table.groupby(['class','split']).size().unstack().reindex(NAMES).plot.bar(ax=axes[0], rot=15)
    axes[0].set(ylabel='Personas', title='Distribucion por clase', xlabel='Clase')
    axes[0].set_xticklabels(display_names,rotation=20,ha='right')
    axes[1].boxplot([table.loc[table['class']==name,'area_fraction'] for name in NAMES], tick_labels=display_names,
                    showfliers=False)
    axes[1].set(ylabel='Area de caja / area de imagen', title='Tamano de las personas')
    plt.setp(axes[1].get_xticklabels(),rotation=20,ha='right')
    fig.savefig(output/'distribution.png', dpi=160)
    plt.close(fig)
    panels, used = [], set()
    for cls in range(len(NAMES)):
        for row in candidates[cls]:
            if row['image'] in used:
                continue
            image = cv2.imread(str(dataset/row['image']))
            draw_scale=max(1.,max(image.shape[:2])/640)
            stroke=max(1,round(draw_scale))
            for line in (dataset/row['label']).read_text().splitlines():
                c,x,y,w,h = map(float,line.split())
                x1,x2 = [round(v*row['width']) for v in (x-w/2,x+w/2)]
                y1,y2 = [round(v*row['height']) for v in (y-h/2,y+h/2)]
                color = [(40,40,220),(0,180,255),(255,160,0),(0,180,0)][int(c)]
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2*stroke)
                cv2.putText(image,NAMES[int(c)],(x1,max(round(15*draw_scale),y1-round(4*draw_scale))),
                            cv2.FONT_HERSHEY_SIMPLEX,.5*draw_scale,color,stroke)
            gain=320/max(image.shape[:2])
            resized=cv2.resize(image,(round(image.shape[1]*gain),round(image.shape[0]*gain)))
            panel=np.full((320,320,3),114,np.uint8)
            top=(320-resized.shape[0])//2;left=(320-resized.shape[1])//2
            panel[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
            panels.append(panel)
            used.add(row['image'])
            if len(panels) == (cls+1)*3:
                break
    cv2.imwrite(str(output/'annotations_train.jpg'),np.vstack([np.hstack(panels[i:i+3]) for i in range(0,3*len(NAMES),3)]))
    print(summary.to_string())
    return report, table


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/ppe_four')
    analyze(parser.parse_args().dataset)
