"""Actual saved-media duration; never infer an incident's end from a short clip."""
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

VERSION=1
FIELDS=('clip_duration_s','clip_duration_source','clip_file_bytes','clip_file_mtime_s','clip_metadata_version')


def safe_path(path,data_dir):
    root=Path(data_dir).resolve();path=Path(path)
    resolved=(root/path if not path.is_absolute() else path).resolve()
    if not resolved.is_relative_to(root/'runs'):
        raise ValueError('Clip must resolve inside the selected data/runs directory')
    return resolved


def atoms(stream,start,end):
    position=start
    for _ in range(10000):
        if position==end:return
        if position+8>end:raise ValueError('Truncated MP4 atom')
        stream.seek(position);header=stream.read(8)
        if len(header)!=8:raise ValueError('Truncated MP4 header')
        size,kind=struct.unpack('>I4s',header);length=8
        if size==1:
            extra=stream.read(8)
            if len(extra)!=8:raise ValueError('Truncated MP4 large atom')
            size=struct.unpack('>Q',extra)[0];length=16
        elif size==0:size=end-position
        if size<length or position+size>end:raise ValueError('Invalid MP4 atom size')
        yield kind,position+length,position+size
        position+=size
    raise ValueError('Too many MP4 atoms')


def movie_duration(path,size):
    """mvhd is presentation duration, not a CFR/frame-count approximation."""
    with path.open('rb') as stream:
        for kind,start,end in atoms(stream,0,size):
            if kind!=b'moov':continue
            seconds=None;video=False
            for child,a,b in atoms(stream,start,end):
                if child==b'mvhd':
                    stream.seek(a);data=stream.read(min(32,b-a))
                    if len(data)<20:raise ValueError('Truncated movie duration')
                    if data[0]==0:scale,ticks=struct.unpack('>II',data[12:20]);unknown=2**32-1
                    elif data[0]==1 and len(data)>=32:scale,ticks=struct.unpack('>IQ',data[20:32]);unknown=2**64-1
                    else:raise ValueError('Unsupported movie-header version')
                    if scale and 0<ticks<unknown:seconds=ticks/scale
                elif child==b'trak':
                    for section,c,d in atoms(stream,a,b):
                        if section!=b'mdia':continue
                        for item,e,f in atoms(stream,c,d):
                            if item==b'hdlr' and f-e>=12:
                                stream.seek(e+8);video=video or stream.read(4)==b'vide'
            if video and seconds is not None and math.isfinite(seconds):return seconds
    return None


def probe_clip(path,*,data_dir):
    """Bounded local-file probe. Missing/invalid/inaccessible media has no duration."""
    try:
        path=safe_path(path,data_dir);stat=path.stat()
        if not path.is_file() or stat.st_size<16:return None
        try:duration=movie_duration(path,stat.st_size)
        except (ValueError,struct.error):duration=None
        method='mp4_mvhd'
        if duration is None:
            ffprobe=shutil.which('ffprobe')
            if not ffprobe:return None
            result=subprocess.run([ffprobe,'-v','error','-select_streams','v:0',
                '-show_entries','format=duration:stream=codec_type','-of','json',str(path)],
                capture_output=True,timeout=3,check=False)
            if result.returncode:return None
            value=json.loads(result.stdout)
            if not any(row.get('codec_type')=='video' for row in value.get('streams',[])):return None
            duration=float(value.get('format',{}).get('duration','nan'));method='ffprobe_format_duration'
        if not math.isfinite(duration) or duration<=0:return None
        # Reject a file replaced/changed while probing it.
        after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):return None
        return {'clip_duration_s':round(duration,6),'clip_duration_source':method,
            'clip_file_bytes':stat.st_size,'clip_file_mtime_s':stat.st_mtime,'clip_metadata_version':VERSION}
    except (OSError,ValueError,TypeError,subprocess.SubprocessError):return None


def cached_clip_metadata(payload,path,*,data_dir):
    try:
        path=safe_path(path,data_dir);stat=path.stat();duration=payload.get('clip_duration_s')
        return (isinstance(duration,(int,float)) and not isinstance(duration,bool) and math.isfinite(duration) and duration>0
            and payload.get('clip_metadata_version')==VERSION and payload.get('clip_file_bytes')==stat.st_size
            and payload.get('clip_file_mtime_s')==stat.st_mtime)
    except (OSError,ValueError,TypeError):return False
