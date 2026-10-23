const {test}=require('node:test');
const assert=require('node:assert/strict');
const {createCameraPreview}=require('../app/static/camera-preview.js');
function fakeStream(){const track={stops:0,stop(){this.stops++}};return {track,getTracks:()=>[track]}}
function fixture(getUserMedia,play=async()=>{}){
  const video={srcObject:null,play,pause(){}};
  const controller=createCameraPreview({getMediaDevices:()=>({getUserMedia,enumerateDevices:async()=>[]}),video});
  return {controller,video};
}
test('preview can release the camera without creating a live session',async()=>{
  const stream=fakeStream();const {controller,video}=fixture(async()=>stream);
  await controller.start();assert.equal(video.srcObject,stream);
  controller.stop();assert.equal(stream.track.stops,1);assert.equal(video.srcObject,null);assert.equal(controller.state.pending,false);
});
test('disconnect while permission request is pending stops a late stream',async()=>{
  let resolve;const stream=fakeStream();const {controller,video}=fixture(()=>new Promise(r=>resolve=r));
  const pending=controller.start();assert.equal(controller.state.pending,true);
  controller.stop();resolve(stream);assert.equal(await pending,null);
  assert.equal(stream.track.stops,1);assert.equal(video.srcObject,null);assert.equal(controller.state.stream,null);
});
test('failed preview playback releases the acquired device',async()=>{
  const stream=fakeStream();const {controller,video}=fixture(async()=>stream,async()=>{throw Error('play failed')});
  await assert.rejects(controller.start(),/play failed/);
  assert.equal(stream.track.stops,1);assert.equal(video.srcObject,null);
});
test('replacing a pending preview never attaches its late stream',async()=>{
  let resolve;const first=fakeStream(),second=fakeStream();let count=0;
  const {controller,video}=fixture(()=>++count===1?new Promise(r=>resolve=r):Promise.resolve(second));
  const pending=controller.start();await controller.start();resolve(first);
  assert.equal(await pending,null);assert.equal(first.track.stops,1);assert.equal(video.srcObject,second);
  controller.stop();assert.equal(second.track.stops,1);
});
