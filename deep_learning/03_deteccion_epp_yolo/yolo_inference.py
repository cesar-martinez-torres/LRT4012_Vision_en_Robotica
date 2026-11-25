"""YOLO11 ONNX inference for four mutually exclusive person PPE states; BGR input."""

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(bgr,size=320):
    height,width=bgr.shape[:2]
    gain=min(size/width,size/height)
    resized=cv2.resize(bgr,(max(1,round(width*gain)),max(1,round(height*gain))),interpolation=cv2.INTER_LINEAR)
    dh,dw=(size-resized.shape[0])/2,(size-resized.shape[1])/2
    top,left=round(dh-0.1),round(dw-0.1)
    padded=cv2.copyMakeBorder(resized,top,round(dh+0.1),left,round(dw+0.1),cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.0
    # Match the native runtime's scaling after each resized side is rounded.
    gains=(resized.shape[1]/width,resized.shape[0]/height)
    return tensor,gains,(left,top)


def decode(raw,gain,padding,shape,confidence=.35,iou=.45):
    gain_x,gain_y= gain if isinstance(gain,(tuple,list)) else (gain,gain)
    predictions=np.asarray(raw)[0].T
    scores=predictions[:,4:].max(1)
    classes=predictions[:,4:].argmax(1)
    selected=scores>=confidence
    xywh,scores,classes=predictions[selected,:4].copy(),scores[selected],classes[selected]
    if not len(scores):
        return []
    xywh[:,:2]-=xywh[:,2:]/2
    xywh[:,0]=(xywh[:,0]-padding[0])/gain_x
    xywh[:,1]=(xywh[:,1]-padding[1])/gain_y
    xywh[:,2]/=gain_x
    xywh[:,3]/=gain_y
    detections=[]
    for cls in np.unique(classes):
        indices=np.flatnonzero(classes==cls)
        keep=cv2.dnn.NMSBoxes(xywh[indices].tolist(),scores[indices].tolist(),confidence,iou)
        for local in np.asarray(keep,dtype=int).reshape(-1):
            index=indices[local]
            x,y,w,h=xywh[index]
            x1,y1=max(0,float(x)),max(0,float(y))
            x2,y2=min(shape[1],float(x+w)),min(shape[0],float(y+h))
            if x2>x1 and y2>y1:
                detections.append({'class_id':int(cls),'confidence':float(scores[index]),'box':[x1,y1,x2,y2]})
    return sorted(detections,key=lambda detection:-detection['confidence'])


class PPEDetector:
    def __init__(self,path,confidence=.35,threads=2):
        path=Path(path)
        self.metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        self.size=self.metadata['input_size']
        self.confidence=confidence
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads
        self.session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])

    def predict(self,bgr):
        tensor,gain,padding=letterbox(bgr,self.size)
        started=perf_counter()
        raw=self.session.run(None,{'images':tensor})[0]
        elapsed=(perf_counter()-started)*1000
        return decode(raw,gain,padding,bgr.shape,self.confidence),elapsed
