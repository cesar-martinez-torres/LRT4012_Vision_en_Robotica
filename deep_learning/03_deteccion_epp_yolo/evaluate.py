"""Select a confidence threshold on validation and evaluate the held-out test set."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from prepare_dataset import ROOT, NAMES, write_json
from train import evaluate as evaluate_ap, sha256


def selected_checkpoint(run):
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    checkpoint=run/selection['selected_run']/'weights/best.pt'
    if sha256(checkpoint)!=selection['checkpoint_sha256']:
        raise ValueError('The selected checkpoint changed.')
    return checkpoint


def overlap(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(intersection/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection+1e-9))


def predict_rows(checkpoint,dataset,split,size,device):
    model=YOLO(checkpoint)
    rows=[r for r in json.loads((dataset/'manifest.json').read_text()) if r['split']==split]
    result=[]
    predictions=model.predict([str(dataset/r['image']) for r in rows],stream=True,batch=8,
                              imgsz=size,rect=False,conf=.05,iou=.45,device=device,verbose=False)
    for row,prediction in zip(rows,predictions):
        truth=[]
        for line in (dataset/row['label']).read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            truth.append([int(cls),(x-w/2)*row['width'],(y-h/2)*row['height'],
                          (x+w/2)*row['width'],(y+h/2)*row['height']])
        boxes=prediction.boxes
        detections=[{'class_id':int(cls),'confidence':float(score),'box':box.tolist()}
                    for cls,score,box in zip(boxes.cls.cpu(),boxes.conf.cpu(),boxes.xyxy.cpu())]
        result.append({'image':row['image'],'truth':truth,'detections':detections})
    return result


def operating_point(rows,confidence):
    matrix=np.zeros((len(NAMES)+1,len(NAMES)+1),dtype=np.int64)
    for row in rows:
        available=set(range(len(row['truth'])))
        for detection in sorted(row['detections'],key=lambda item:-item['confidence']):
            if detection['confidence']<confidence:
                continue
            candidates=[i for i in available if overlap(row['truth'][i][1:],detection['box'])>=.5]
            if candidates:
                index=max(candidates,key=lambda i:overlap(row['truth'][i][1:],detection['box']))
                matrix[row['truth'][index][0],detection['class_id']]+=1
                available.remove(index)
            else:
                matrix[len(NAMES),detection['class_id']]+=1
        for index in available:
            matrix[row['truth'][index][0],len(NAMES)]+=1
    classes=[]
    for i,name in enumerate(NAMES):
        tp=int(matrix[i,i]);fp=int(matrix[:,i].sum()-tp);fn=int(matrix[i,:].sum()-tp)
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        classes.append({'class':name,'precision':precision,'recall':recall,
                        'f1':2*tp/max(2*tp+fp+fn,1),'tp':tp,'fp':fp,'fn':fn})
    return {'confidence':confidence,'nms_iou':.45,'matching_iou':.5,
            'matching':'Greedy confidence order, maximum spatial IoU, class-agnostic matching.',
            'images':len(rows),'matrix_reference_rows_predicted_columns':matrix.tolist(),
            'per_class':classes,'macro_f1':float(np.mean([c['f1'] for c in classes])),
            'note':'Background is unmatched detections/objects, not a learned class; background/background is not a true-negative count.'}


def plot_matrix(report,path):
    matrix=np.asarray(report['matrix_reference_rows_predicted_columns'])
    names=['Persona','Solo casco','Solo chaleco','Casco y chaleco','Fondo']
    fig,ax=plt.subplots(figsize=(10,8),constrained_layout=True)
    ax.imshow(matrix,cmap='Greys')
    for (y,x),value in np.ndenumerate(matrix):
        ax.text(x,y,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black')
    ax.set(xticks=range(len(NAMES)+1),yticks=range(len(NAMES)+1),xticklabels=names,yticklabels=names,
           xlabel='Prediccion',ylabel='Referencia',title=f'confidence={report["confidence"]:.2f}; IoU=0.50')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/yolo11n_ppe')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/ppe_four')
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=('0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    checkpoint=selected_checkpoint(args.run)
    recipe=json.loads((args.run/'recipe.json').read_text())
    destination=args.run/f'operating_{args.split}.json'
    if destination.exists():
        raise FileExistsError(f'Evaluation already exists: {destination}')
    threshold_path=args.run/'operating_selection.json'
    if args.split=='test' and not threshold_path.is_file():
        raise ValueError('Select the confidence threshold on validation before testing.')
    rows=predict_rows(checkpoint,args.dataset,args.split,recipe['imgsz'],device)
    if args.split=='val':
        trials=[operating_point(rows,round(float(value),2)) for value in np.arange(.10,.76,.05)]
        selected=max(trials,key=lambda item:(item['macro_f1'],item['confidence']))
        write_json(threshold_path,{'selected_at_utc':datetime.now(timezone.utc).isoformat(),
            'criterion':'Maximum macro-F1 on validation; ties choose higher confidence.',
            'confidence':selected['confidence'],'checkpoint_sha256':sha256(checkpoint),
            'trials':[{'confidence':r['confidence'],'macro_f1':r['macro_f1']} for r in trials]})
        report=selected
    else:
        threshold=json.loads(threshold_path.read_text())
        if threshold['checkpoint_sha256']!=sha256(checkpoint):
            raise ValueError('Threshold selection belongs to a different checkpoint.')
        evaluate_ap(checkpoint,args.dataset/'dataset.yaml',args.run,'test',recipe['imgsz'],8,device)
        report=operating_point(rows,threshold['confidence'])
    report.update(split=args.split,checkpoint_sha256=sha256(checkpoint))
    write_json(args.run/f'predictions_{args.split}.json',rows)
    write_json(destination,report)
    plot_matrix(report,args.run/f'operating_{args.split}_matrix.png')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
