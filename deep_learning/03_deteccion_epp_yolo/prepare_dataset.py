"""Prepare person-state labels from explicit helmet/no_helmet and vest/no_vest annotations."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image
import requests

ROOT=Path(__file__).resolve().parent
NAMES=['persona','persona_con_casco','persona_con_chaleco','persona_epp_completo']
URL='https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip'
ARCHIVE_SHA256='e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232'

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def download(url,path):
    path=Path(path)
    if path.is_file():return
    path.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(url,stream=True,timeout=120) as response:
        response.raise_for_status()
        with path.open('wb') as handle:
            for chunk in response.iter_content(1024*1024):handle.write(chunk)

def ensure_pretrained():
    path=ROOT/'data/pretrained/yolo11n.pt'
    download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1':
        raise ValueError('Unexpected pretrained checkpoint checksum.')
    return path

def associate(text):
    a=np.asarray([list(map(float,l.split())) for l in text.splitlines()],dtype=float).reshape(-1,5)
    c=a[:,0].astype(int);boxes=np.c_[a[:,1:3]-a[:,3:]/2,a[:,1:3]+a[:,3:]/2]
    people=np.flatnonzero(c==5)
    assigned={int(p):{0:set(),1:set()} for p in people}
    for j in np.flatnonzero(np.isin(c,[0,6,2,11])):
        area=np.prod(boxes[j,2:]-boxes[j,:2]);center=(boxes[j,:2]+boxes[j,2:])/2
        candidates=[]
        for p in people:
            b=boxes[p];overlap=np.prod(np.maximum(0,np.minimum(b[2:],boxes[j,2:])-np.maximum(b[:2],boxes[j,:2])))/max(area,1e-10)
            relative_y=(center[1]-b[1])/max(b[3]-b[1],1e-10)
            is_head=c[j] in (0,6)
            if overlap>=.8 and b[0]<=center[0]<=b[2] and (0<=relative_y<=.55 if is_head else .1<=relative_y<=.85):
                candidates.append((np.prod(b[2:]-b[:2]),int(p)))
        candidates.sort()
        # Ambiguous overlapping bodies are excluded instead of assigning equipment twice.
        if len(candidates)>1 and candidates[0][0]>.75*candidates[1][0]:
            for _,p in candidates:assigned[p][0 if c[j] in (0,6) else 1].add(-1)
        elif candidates:
            assigned[candidates[0][1]][0 if c[j] in (0,6) else 1].add(int(c[j]))
    labels=[];details=[]
    for p in people:
        head,torso=assigned[int(p)][0],assigned[int(p)][1]
        if len(head)!=1 or len(torso)!=1 or -1 in head or -1 in torso:return None
        state=int(0 in head)+2*int(2 in torso)
        x1,y1,x2,y2=np.clip(boxes[p],0,1)
        if min(x2-x1,y2-y1)<=0:return None
        labels.append(f'{state} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}')
        details.append({'source_person_row':int(p),'helmet_annotation':next(iter(head)),
                        'vest_annotation':next(iter(torso)),'state':state})
    return (labels,details) if len(people) else None

def prepare(archive_path,output):
    if output.exists():raise FileExistsError(f'Dataset exists: {output}; use --audit.')
    if hashlib.sha256(archive_path.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError('Source checksum changed.')
    records=[];excluded=[]
    with zipfile.ZipFile(archive_path) as archive:
        for label_path in sorted(n for n in archive.namelist() if '/labels/' in n and n.endswith('.txt')):
            association=associate(archive.read(label_path).decode())
            image_path=label_path.replace('/labels/','/images/').removesuffix('.txt')+'.jpg'
            if association is None:
                excluded.append({'source':image_path,'reason':'No people or at least one ambiguous/incomplete explicit head/torso state.'});continue
            content=archive.read(image_path)
            end=content.rfind(b'\xff\xd9')
            if end>=0:content=content[:end+2]
            with Image.open(BytesIO(content)) as image:
                width,height=image.size;rgb=np.asarray(image.convert('RGB'))
                gray=np.asarray(image.convert('L').resize((32,32)),dtype=np.float32)
            dct=cv2.dct(gray)[:8,:8].flatten()[1:]
            phash=int(''.join('1' if v>np.median(dct) else '0' for v in dct),2)
            records.append({'name':Path(image_path).name,'original_split':Path(label_path).parent.name,
                'width':width,'height':height,'content':content,'gray':gray,'phash':phash,
                'pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                'labels':association[0],'associations':association[1]})
        parent=list(range(len(records)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        pairs=0
        for i,row in enumerate(records):
            for j in range(i):
                other=records[j]
                exact=row['pixel_sha256']==other['pixel_sha256']
                near=((row['phash']^other['phash']).bit_count()<=6 and
                    cv2.matchTemplate(row['gray'],other['gray'],cv2.TM_CCOEFF_NORMED)[0,0]>=.98 and
                    np.abs(row['gray']-other['gray']).mean()<=12)
                if exact or near:parent[find(i)]=find(j);pairs+=1
        groups=defaultdict(list)
        for i in range(len(records)):groups[find(i)].append(i)
        # Fixed group split, selected only for annotation balance, before any training.
        group_list=list(groups.values());hist=np.array([[sum(int(l.split()[0])==c for i in g for l in records[i]['labels']) for c in range(4)] for g in group_list])
        target=np.array([.7,.15,.15]);best=None
        for seed in range(4012,4212):
            order=np.random.default_rng(seed).permutation(len(group_list));n=len(order)
            partitions=np.split(order,[round(.7*n),round(.85*n)])
            totals=np.array([hist[p].sum(axis=0) for p in partitions])
            score=np.square(totals/hist.sum(axis=0)-target[:,None]).sum()
            if best is None or score<best[0]:best=(score,seed,partitions)
        output.mkdir(parents=True);manifest=[];seen=set()
        for split,indices in zip(('train','val','test'),best[2]):
            for g in indices:
                group_id=min(records[i]['name'] for i in group_list[g])
                for i in group_list[g]:
                    r=records[i]
                    if r['pixel_sha256'] in seen:continue
                    seen.add(r['pixel_sha256'])
                    img=Path('images')/split/r['name'];lab=Path('labels')/split/Path(r['name']).with_suffix('.txt')
                    (output/img).parent.mkdir(parents=True,exist_ok=True);(output/lab).parent.mkdir(parents=True,exist_ok=True)
                    (output/img).write_bytes(r['content']);(output/lab).write_text('\n'.join(r['labels'])+'\n')
                    manifest.append({k:r[k] for k in ('width','height','pixel_sha256','original_split','associations')}|{
                        'image':img.as_posix(),'label':lab.as_posix(),'split':split,'source_name':r['name'],'group_id':group_id})
        for name in ('LICENSE.txt','README.md'):(output/('SOURCE_'+name)).write_bytes(archive.read('construction-ppe/'+name))
    write_json(output/'manifest.json',manifest);write_json(output/'excluded_images.json',excluded)
    (output/'dataset.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(NAMES)),encoding='utf-8')
    write_json(output/'provenance.json',{'source':'Ultralytics Construction-PPE, November 2026','download':URL,
        'archive_sha256':ARCHIVE_SHA256,'source_images':1000,'resolved_images_before_dedup':len(records),
        'excluded_images':len(excluded),'duplicate_pairs':pairs,'split_seed':best[1],
        'license':'CC BY 4.0; Ultralytics team; adapted annotations to four person states.',
        'data_type':'Mixed human-collected and generated images; source provides no per-image origin flag.',
        'derivation':'Explicit helmet/no_helmet and vest/no_vest body-region annotations assigned to enclosing person; no missing-label inference. Entire ambiguous images excluded.',
        'split_policy':'70/15/15 by conservative duplicate group, balanced using labels only. Source has no test set; this is a new internal holdout.',
        'limitations':'Derived state labels inherit annotation and association errors. Scene/session independence is not guaranteed. Complete EPP means helmet and vest visibly present only.'})
    return audit(output)

def audit(output):
    rows=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    counts={s:{'images':0,'empty_images':0,'objects':[0]*4} for s in ('train','val','test')};hashes={};groups={}
    for row in rows:
        split=row['split'];counts[split]['images']+=1
        with Image.open(output/row['image']) as image:
            if image.size!=(row['width'],row['height']):raise ValueError('Image dimensions changed.')
            digest=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
        if digest!=row['pixel_sha256']:raise ValueError('Image pixels changed.')
        for table,key in ((hashes,digest),(groups,row['group_id'])):
            if key in table and table[key]!=split:raise ValueError('Cross-split duplicate.')
            table[key]=split
        for line in (output/row['label']).read_text().splitlines():
            c,x,y,w,h=map(float,line.split())
            if c!=int(c) or not 0<=c<4 or not np.isfinite([x,y,w,h]).all() or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:raise ValueError('Invalid box.')
            counts[split]['objects'][int(c)]+=1
    if any(min(c['objects'])<=0 for c in counts.values()):raise ValueError('Missing state in a partition.')
    report={'status':'PASS','classes':NAMES,'partitions':counts,'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
        'cross_split_exact_pixel_duplicates':0,'cross_split_group_duplicates':0,'scene_independence':'Not guaranteed.'}
    write_json(output/'audit_report.json',report);print(json.dumps(report,indent=2),flush=True);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',action='store_true')
    parser.add_argument('--archive',type=Path,default=ROOT/'data/raw/construction-ppe.zip');args=parser.parse_args()
    if args.audit:audit(ROOT/'data/ppe_four')
    else:
        download(URL,args.archive);prepare(args.archive,ROOT/'data/ppe_four');ensure_pretrained()
