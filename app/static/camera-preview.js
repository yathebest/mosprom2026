/* Camera ownership is separate from an inference session. */
(function(root){
  function createCameraPreview({getMediaDevices,video,onChange=()=>{}}){
    let stream=null,pending=false,generation=0;
    const notify=()=>onChange({stream,pending});
    function stop(){
      generation++;pending=false;
      if(stream)stream.getTracks().forEach(track=>track.stop());
      stream=null;video.pause();video.srcObject=null;notify();
    }
    async function start(deviceId=''){
      stop();const request=generation;pending=true;notify();
      try{
        const devices=getMediaDevices();
        if(!devices?.getUserMedia)throw Error('Браузер не предоставляет камеру. Откройте localhost в Chrome или Edge.');
        const acquired=await devices.getUserMedia({video:{width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:25},...(deviceId?{deviceId:{exact:deviceId}}:{})},audio:false});
        if(request!==generation){acquired.getTracks().forEach(track=>track.stop());return null}
        stream=acquired;video.srcObject=stream;notify();await video.play();
        if(request!==generation)return null;
        const available=await devices.enumerateDevices();
        if(request!==generation)return null;
        pending=false;notify();return {stream,devices:available};
      }catch(error){
        if(request!==generation)return null;
        stop();throw error;
      }
    }
    return {start,stop,get state(){return {stream,pending}}};
  }
  if(typeof module==='object'&&module.exports)module.exports={createCameraPreview};
  else root.createCameraPreview=createCameraPreview;
})(globalThis);
