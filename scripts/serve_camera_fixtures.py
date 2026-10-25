"""Two loopback-only MJPEG test sources. Replays source pixels; never fakes inference."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2


def frames(path, name, width=640, maximum=250):
    capture=cv2.VideoCapture(str(path)); result=[]
    source_fps=capture.get(cv2.CAP_PROP_FPS)
    if not 1<=source_fps<=120:source_fps=25.
    try:
        while len(result)<maximum:
            ok,image=capture.read()
            if not ok:break
            scale=min(1.,width/image.shape[1]); image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
            # Distinct, explicit labels prevent mistaking these sources for physical cameras.
            cv2.rectangle(image,(0,0),(image.shape[1],27),(15,40,42),-1)
            cv2.putText(image,'LOCAL TEST REPLAY - '+name,(9,19),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,245,235),1,cv2.LINE_AA)
            ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,82])
            if ok:result.append(jpeg.tobytes())
    finally:capture.release()
    if not result:raise ValueError('Cannot decode test source: '+str(path))
    return result,source_fps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a',type=Path,required=True);parser.add_argument('--b',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8766);parser.add_argument('--fps',type=float,default=12)
    args=parser.parse_args();cv2.setNumThreads(1)
    sources={'/cam-a.mjpg':frames(args.a,'CAMERA A'),'/cam-b.mjpg':frames(args.b,'CAMERA B')}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in sources:self.send_error(404);return
            images,fps=sources[self.path]
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store');self.send_header('X-Test-Source','Known-dataset replay; not a physical camera')
            self.end_headers();origin=time.monotonic();due=origin
            try:
                while True:
                    now=time.monotonic();image=images[int((now-origin)*fps)%len(images)]
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+b'\r\n\r\n'+image+b'\r\n');self.wfile.flush()
                    due+=1/args.fps;time.sleep(max(0.,due-time.monotonic()))
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True
    print(f'Test replays: http://127.0.0.1:{args.port}/cam-a.mjpg and /cam-b.mjpg',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
