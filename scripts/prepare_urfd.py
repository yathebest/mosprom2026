"""Fetch reproducible URFD RGB evaluation clips, not depth composites.

Default: four samples. Use --all for all cam0 previews linked by the authors.
The previews are low-resolution; original RGB PNG archives are preferable
for serious model evaluation. Preserve the accompanying license/attribution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen
import re

import cv2

PAGE = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=="a":
            self.links.extend(value for key,value in attrs if key=="href" and value)


def fetch(url):
    with urlopen(url,timeout=40) as response:
        data=response.read(50*1024*1024+1)
    if len(data)>50*1024*1024:
        raise ValueError("Unexpectedly large preview")
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    links=Links();links.feed(fetch(PAGE).decode())
    urls=sorted({urljoin(PAGE,h) for h in links.links if re.search(r"/(fall|adl)-\d{2}-cam0\.mp4$",h)})
    if not args.all:
        urls=[u for u in urls if re.search(r"(fall|adl)-0[12]-cam0\.mp4$",u)]
    target=ROOT/"data"/"datasets"/"urfd"
    target.mkdir(parents=True,exist_ok=True)
    for annotation in ('urfall-cam0-falls.csv','urfall-cam0-adls.csv'):
        path=target/annotation
        if not path.exists():
            path.write_bytes(fetch(urljoin(PAGE,'data/'+annotation)))

    def one(url):
        name=url.rsplit("/",1)[1]
        original=target/name
        if not original.exists():
            original.write_bytes(fetch(url))
        cap=cv2.VideoCapture(str(original))
        w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or w%2 or not fps:
            raise ValueError("Invalid preview "+name)
        rgb=target/(original.stem+"-rgb.mp4")
        writer=cv2.VideoWriter(str(rgb),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w//2,h))
        count=0
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            writer.write(frame[:,w//2:]);count+=1
        cap.release();writer.release()
        print(name,count,"frames",flush=True)
        return {"source":url,"original":name,"rgb":rgb.name,"frames":count,"fps":fps,
                "sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
                "transform":"Right half of authors' depth+RGB preview; MP4V re-encode",
                "split_unit":original.stem,"scenario":"fall" if name.startswith("fall") else "ADL"}

    with ThreadPoolExecutor(max_workers=4) as executor:
        manifest=list(executor.map(one,urls))
    (target/"manifest.json").write_text(json.dumps({
        "dataset":"UR Fall Detection","authors":"Bogdan Kwolek, Michal Kepski",
        "citation":"Human fall detection on embedded platform using depth maps and wireless accelerometer, 2014",
        "page":PAGE,"license":"CC BY-NC-SA 4.0","files":manifest,
        "note":"Scenario names are not frame-level event annotations. Do not report event precision/recall from names alone."},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
