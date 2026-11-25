"""Run the PPE ONNX detector on images, video or a USB camera."""

import argparse
import json
from itertools import count
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np

from yolo_inference import PPEDetector

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT if (ROOT/'model.onnx').is_file() else ROOT/'runs/pc_ppe'


def annotate(image,detections,names):
    view=image.copy()
    colors=[(40,40,220),(0,180,255),(255,160,20),(40,200,70)]
    scale=max(1.,max(image.shape[:2])/640)
    stroke=max(1,round(scale))
    for detection in detections:
        x1,y1,x2,y2=map(round,detection['box']);cls=detection['class_id']
        color=colors[cls%len(colors)]
        cv2.rectangle(view,(x1,y1),(x2,y2),color,2*stroke)
        cv2.putText(view,f'{names[cls]} {detection["confidence"]:.2f}',(x1,max(round(18*scale),y1-round(5*scale))),
                    cv2.FONT_HERSHEY_SIMPLEX,.5*scale,color,stroke,cv2.LINE_AA)
    return view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=BUNDLE/'model.onnx')
    parser.add_argument('--source',default=str(BUNDLE/'samples'),help='Image, folder, video path or webcam index (0, 1, ...).')
    parser.add_argument('--output',type=Path,default=ROOT/'results' if BUNDLE==ROOT else ROOT/'runs/onnx_check')
    parser.add_argument('--confidence',type=float,help='Default: threshold chosen on validation.')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-frames',type=int,default=0,help='Maximum frames; 0 processes all images/video or runs webcam until Q/Esc/Ctrl+C.')
    parser.add_argument('--show',action='store_true',help='Images: any key advances, Q exits. Webcam/video: Q exits.')
    args=parser.parse_args()
    if args.max_frames<0:parser.error('--max-frames must be zero or positive.')
    if args.threads<=0:parser.error('--threads must be positive.')
    if args.confidence is not None and not 0<args.confidence<=1:parser.error('--confidence must be in (0, 1].')
    cv2.setNumThreads(1)
    metadata=json.loads(args.model.with_suffix('.json').read_text())
    confidence=metadata['confidence'] if args.confidence is None else args.confidence
    detector=PPEDetector(args.model,confidence,args.threads)
    path=Path(args.source)
    suffixes={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
    images=sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes) if path.is_dir() else ([path] if path.suffix.lower() in suffixes else [])
    if path.is_dir() and not images:
        raise ValueError('The folder contains no supported images.')
    capture=None
    if not images:
        source=int(args.source) if args.source.isdecimal() else args.source
        capture=cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f'Cannot open camera/video: {args.source}')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.show:
        cv2.namedWindow('Estado de EPP por persona',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Estado de EPP por persona',960,720)
    timings=[];started=perf_counter()
    try:
        with (args.output/'detections.jsonl').open('w',encoding='utf-8') as handle:
            for index in count():
                if args.max_frames and index>=args.max_frames:break
                if images:
                    if index>=len(images):break
                    image=cv2.imread(str(images[index]))
                    source_name=images[index].name
                    if image is None:raise ValueError(f'Cannot decode {images[index]}')
                else:
                    valid,image=capture.read()
                    if not valid:break
                    source_name=str(index)
                processing_started=perf_counter()
                detections,inference_ms=detector.predict(image)
                view=annotate(image,detections,metadata['classes'])
                processing_ms=(perf_counter()-processing_started)*1000
                timings.append({'inference_ms':inference_ms,'processing_ms':processing_ms})
                handle.write(json.dumps({'frame':index,'source':source_name,'objects':detections,**timings[-1]})+'\n')
                if images or index%30==0:
                    cv2.imwrite(str(args.output/f'frame_{index:06d}.jpg'),view)
                if args.show:
                    cv2.imshow('Estado de EPP por persona',view)
                    # Still images remain visible; live capture continues between keys.
                    if cv2.waitKey(0 if images else 1)&255 in (ord('q'),ord('Q'),27):break
                if index%30==0:print(f'frame={index} objects={len(detections)} inference_ms={inference_ms:.1f}',flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if capture is not None:capture.release()
        if args.show:cv2.destroyAllWindows()
    if not timings:raise RuntimeError('No frames processed.')
    stable=timings[min(5,len(timings)-1):]
    report={'platform':platform.platform(),'machine':platform.machine(),'threads':args.threads,
        'classes':metadata['classes'],'input_size':detector.size,'confidence':confidence,
        'frames':len(timings),'end_to_end_fps':len(timings)/(perf_counter()-started),
        'median_inference_ms_after_warmup':float(np.median([r['inference_ms'] for r in stable])),
        'median_processing_ms_after_warmup':float(np.median([r['processing_ms'] for r in stable])),
        'note':'Inference is ONNX only; processing includes resize, normalization, NMS and annotation. This measurement applies only to this computer.'}
    (args.output/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
