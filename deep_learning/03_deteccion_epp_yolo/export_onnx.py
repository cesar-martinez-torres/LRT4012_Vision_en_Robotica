"""Export selected YOLO weights and compare ONNX/PyTorch on validation images."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

from prepare_dataset import ROOT, NAMES, write_json
from evaluate import selected_checkpoint, overlap
from train import sha256
from yolo_inference import letterbox, decode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/yolo11n_ppe')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/ppe_four')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/pc_ppe')
    parser.add_argument('--samples',type=int,default=24)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Package exists: {args.output}')
    torch.set_num_threads(2)
    checkpoint=selected_checkpoint(args.run)
    recipe=json.loads((args.run/'recipe.json').read_text())
    threshold=json.loads((args.run/'operating_selection.json').read_text())
    if threshold['checkpoint_sha256']!=sha256(checkpoint):
        raise ValueError('Threshold and selected model differ.')
    size=recipe['imgsz']
    model=YOLO(checkpoint)
    exported=Path(model.export(format='onnx',imgsz=size,opset=17,dynamic=False,
                               simplify=False,nms=False,half=False,device='cpu'))
    onnx.checker.check_model(str(exported))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(exported),options,providers=['CPUExecutionProvider'])
    rows=[r for r in json.loads((args.dataset/'manifest.json').read_text()) if r['split']=='val']
    if args.samples < len(NAMES):
        raise ValueError('--samples must cover all four states (at least 4).')
    sampled=[]
    for cls in range(len(NAMES)):
        candidates=[r for r in rows if any(a['state']==cls for a in r['associations'])]
        candidates.sort(key=lambda r:max(float(l.split()[3])*float(l.split()[4]) for l in (args.dataset/r['label']).read_text().splitlines() if int(l.split()[0])==cls),reverse=True)
        candidate=next((r for r in candidates if r not in sampled),candidates[0])
        if candidate not in sampled:sampled.append(candidate)
    for index in np.linspace(0,len(rows)-1,len(rows),dtype=int):
        if len(sampled)>=min(args.samples,len(rows)):break
        if rows[index] not in sampled:sampled.append(rows[index])
    coordinate_error=0.;score_error=0.;detection_count=0;unmatched=0;matched=0;box_error=0.
    model.model.eval()
    for row in sampled:
        bgr=cv2.imread(str(args.dataset/row['image']))
        tensor,gain,padding=letterbox(bgr,size)
        with torch.inference_mode():
            native=model.model(torch.from_numpy(tensor))[0].cpu().numpy()
        converted=session.run(None,{session.get_inputs()[0].name:tensor})[0]
        coordinate_error=max(coordinate_error,float(np.max(np.abs(native[:,:4]-converted[:,:4]))))
        score_error=max(score_error,float(np.max(np.abs(native[:,4:]-converted[:,4:]))))
        detections=decode(converted,gain,padding,bgr.shape,threshold['confidence'])
        detection_count+=len(detections)
        reference=model.predict(bgr,imgsz=size,rect=False,conf=threshold['confidence'],
                                iou=.45,device='cpu',verbose=False)[0].boxes
        available=set(range(len(detections)))
        for box,cls,score in zip(reference.xyxy.cpu().numpy(),reference.cls.cpu().numpy(),reference.conf.cpu().numpy()):
            candidates=[i for i in available if detections[i]['class_id']==int(cls)]
            if not candidates:
                unmatched+=1;continue
            best=max(candidates,key=lambda i:overlap(box,detections[i]['box']))
            if overlap(box,detections[best]['box'])<.99 or abs(float(score)-detections[best]['confidence'])>.001:
                unmatched+=1;continue
            box_error=max(box_error,float(np.max(np.abs(box-detections[best]['box']))))
            matched+=1
            available.remove(best)
        unmatched+=len(available)
    report={'split':'val','images':len(sampled),'detections':detection_count,
            'max_raw_coordinate_error_pixels':coordinate_error,'max_raw_probability_error':score_error,
            'coordinate_tolerance_pixels':.05,'probability_tolerance':.001,
            'unmatched_native_onnx_detections':unmatched,
            'matched_native_onnx_detections':matched,'max_box_error_original_pixels':box_error,
            'status':'PASS' if coordinate_error<.05 and score_error<.001 and detection_count>0 and unmatched==0 else 'FAIL'}
    write_json(args.run/'onnx_validation.json',report)
    if report['status']!='PASS':
        raise RuntimeError(f'ONNX validation failed: {report}')
    args.output.mkdir(parents=True)
    shutil.copy2(exported,args.output/'model.onnx')
    write_json(args.output/'model.json',{'classes':NAMES,'input_size':size,
        'confidence':threshold['confidence'],'color':'RGB','normalization':'divide by 255',
        'resize':'centered letterbox; padding=114','output':'xywh pixels + class probabilities; no objectness',
        'box_rescaling':'per-axis gains after rounding resized width and height',
        'nms':'class-aware; IoU=.45; applied outside ONNX','opset':17,
        'checkpoint_sha256':sha256(checkpoint),'model_sha256':sha256(args.output/'model.onnx')})
    for name in ['yolo_inference.py','infer.py','check_environment.py']:
        shutil.copy2(ROOT/name,args.output/name)
    shutil.copy2(ROOT/'requirements_inference.txt',args.output/'requirements.txt')
    (args.output/'README.md').write_text(
        '# Inferencia de EPP en PC\n\nActivar el entorno de deep learning y ejecutar desde esta carpeta:\n\n'
        '```powershell\npython -m pip install -r requirements.txt\n'
        'python check_environment.py\npython infer.py --source samples --show\n'
        'python infer.py --source 0 --show\n```\n\n'
        'Imagen propia: `python infer.py --source "C:\\ruta\\foto.jpg" --show`. '
        'Q o Esc termina; cualquier tecla avanza en imagenes. '
        'Para otra webcam usar --source 1. Resultados en results.\n\n'
        'Estados: persona sin casco ni chaleco, solo casco, solo chaleco y ambos. '
        'EPP completo significa casco y chaleco visibles; la fuente incluye otras prendas de alta visibilidad.\n',encoding='utf-8')
    shutil.copy2(args.dataset/'SOURCE_LICENSE.txt',args.output/'SOURCE_LICENSE.txt')
    shutil.copy2(args.dataset/'SOURCE_README.md',args.output/'SOURCE_README.md')
    shutil.copy2(ROOT/'FUENTES.md',args.output/'FUENTES.md')
    for name in ['selection.json','operating_selection.json','metrics_test.json','operating_test.json','onnx_validation.json']:
        shutil.copy2(args.run/name,args.output/name)
    (args.output/'samples').mkdir()
    for row in sampled[:8]:
        shutil.copy2(args.dataset/row['image'],args.output/'samples'/Path(row['image']).name)
    archive=shutil.make_archive(str(args.output),'zip',args.output)
    print(json.dumps({'package':str(args.output),'zip':archive,'validation':report},indent=2))


if __name__=='__main__':
    main()
