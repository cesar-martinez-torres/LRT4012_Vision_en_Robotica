"""Plot the training history and inspect test detections and failure examples."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_dataset import ROOT, NAMES
from infer import annotate
from evaluate import operating_point


def analyze(run,dataset):
    output=run/'analysis';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    offset=0
    for stage in ('adaptation','finetune'):
        history=pd.read_csv(run/stage/'results.csv')
        history.columns=history.columns.str.strip()
        x=np.arange(len(history))+offset+1
        axes[0].plot(x,history['metrics/mAP50(B)'],label=f'{stage}: mAP50')
        axes[0].plot(x,history['metrics/mAP50-95(B)'],'--',label=f'{stage}: mAP50-95')
        axes[1].plot(x,history['train/box_loss'],label=f'{stage}: train box')
        axes[1].plot(x,history['val/box_loss'],'--',label=f'{stage}: val box')
        offset+=len(history)
    axes[0].set(xlabel='Accumulated epoch',ylabel='Validation AP',ylim=(0,1))
    axes[1].set(xlabel='Accumulated epoch',ylabel='Box loss')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output/'learning_curves.png',dpi=160);plt.close(fig)
    report=json.loads((run/'metrics_test.json').read_text())
    pd.DataFrame(report['per_class']).to_csv(output/'test_per_class.csv',index=False)
    operating=json.loads((run/'operating_test.json').read_text())
    pd.DataFrame(operating['per_class']).to_csv(output/'test_operating_point.csv',index=False)
    predictions=json.loads((run/'predictions_test.json').read_text())
    failures=[]
    for row in predictions:
        counts=operating_point([row],operating['confidence'])
        errors=sum(c['fp']+c['fn'] for c in counts['per_class'])
        failures.append((errors,row))
    for filename,rows in [('test_examples.jpg',predictions[:9]),
                          ('test_failures.jpg',[r for _,r in sorted(failures,key=lambda item:-item[0])[:9]])]:
        panels=[]
        for row in rows:
            image=cv2.imread(str(dataset/row['image']))
            # Reference in white; detections retain class colors.
            for cls,x1,y1,x2,y2 in row['truth']:
                cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),(255,255,255),1)
            detections=[d for d in row['detections'] if d['confidence']>=operating['confidence']]
            image=annotate(image,detections,NAMES)
            gain=320/max(image.shape[:2])
            resized=cv2.resize(image,(round(image.shape[1]*gain),round(image.shape[0]*gain)))
            panel=np.full((320,320,3),114,np.uint8)
            top=(320-resized.shape[0])//2;left=(320-resized.shape[1])//2
            panel[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
            panels.append(panel)
        cv2.imwrite(str(output/filename),np.vstack([np.hstack(panels[i:i+3]) for i in (0,3,6)]))
    print(json.dumps({'test':report['metrics'],'operating_point':operating['per_class'],
                      'examples':'White: reference; colors: predictions.'},indent=2))
    return report,operating


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/yolo11n_ppe')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/ppe_four')
    args=parser.parse_args();analyze(args.run,args.dataset)
