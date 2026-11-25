"""Verify the ONNX model and CPU runtime before inference."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
import onnxruntime as ort


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    default=root/'model.onnx' if (root/'model.onnx').is_file() else root/'runs/pc_ppe/model.onnx'
    parser.add_argument('--model',type=Path,default=default)
    args=parser.parse_args()
    board=None
    metadata=json.loads(args.model.with_suffix('.json').read_text())
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Model checksum does not match metadata.')
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(args.model),options,providers=['CPUExecutionProvider'])
    size=metadata['input_size']
    raw=session.run(None,{session.get_inputs()[0].name:np.zeros((1,3,size,size),np.float32)})[0]
    if raw.shape[1]!=4+len(metadata['classes']) or not np.isfinite(raw).all():
        raise ValueError('Unexpected output shape or non-finite values.')
    print(json.dumps({'status':'PASS','machine':platform.machine(),'board':board,'onnxruntime':ort.__version__,
        'opencv':cv2.__version__,'numpy':np.__version__,
        'classes':metadata['classes'],'input_size':size,'output_shape':list(raw.shape)},indent=2))


if __name__=='__main__':
    main()
