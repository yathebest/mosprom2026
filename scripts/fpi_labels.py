"""Read FPI's concatenated five-token YOLO records without guessing polygons."""
from pathlib import Path
import math


def parse_labels(text):
    records=[];seen=set()
    for line in text.splitlines():
        values=line.split()
        if not values:continue
        if len(values)%5:raise ValueError('FPI record is not a sequence of five-token boxes')
        for start in range(0,len(values),5):
            label,x,y,w,h=map(float,values[start:start+5])
            if label not in (0,1) or not all(math.isfinite(v) for v in (x,y,w,h)) or not all(0<=v<=1 for v in (x,y,w,h)) or min(w,h)<=0:
                raise ValueError('Invalid FPI class/normalized box')
            row=(int(label),x,y,w,h)
            if row not in seen:records.append(row);seen.add(row)
    return records


def read_labels(path):return parse_labels(Path(path).read_text())
