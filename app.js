// Dots Classifier — TF.js client-side app

const el = id => document.getElementById(id);

let model = null;
let chart = null;
let running = false;
let pauseRequested = false;
let globalPreviewImages = [];

function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

function createImageBuffer(size, dotCount, dotRadius, rng, allowOverlap) {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'black';
  ctx.fillRect(0,0,size,size);
  ctx.fillStyle = 'white';
  const positions = [];
  for (let i=0;i<dotCount;i++){
    let attempts=0;
    while(true){
      attempts++;
      const x = Math.floor(rng() * (size-2*dotRadius)) + dotRadius;
      const y = Math.floor(rng() * (size-2*dotRadius)) + dotRadius;
      let ok = true;
      if(!allowOverlap){
        for(const p of positions){
          const dx = p.x-x; const dy = p.y-y;
          if(Math.hypot(dx,dy) < (dotRadius*2)) { ok=false; break; }
        }
      }
      if(ok){ positions.push({x,y}); break; }
      if(attempts>200) { positions.push({x,y}); break; }
    }
  }
  for(const p of positions) ctx.beginPath(), ctx.arc(p.x,p.y,dotRadius,0,Math.PI*2), ctx.fill();
  const imd = ctx.getImageData(0,0,size,size);
  // return grayscale Float32Array normalized 0..1
  const data = new Float32Array(size*size);
  for(let i=0;i<size*size;i++) data[i] = imd.data[i*4]/255;
  return {data, imageData:imd};
}

function makeGenerator(opts, forValidation=false){
  const {batchSize, size, dotRadius, countMin, countMax, allowOverlap, seed} = opts;
  const rngBase = mulberry32(seed + (forValidation?123456:0));
  return function*(){
    while(true){
      const batchX = new Float32Array(batchSize * size * size);
      const batchY = new Float32Array(batchSize * opts.numClasses);
      let firstImageData = null;
      let firstImageTensor = null;
      let firstClassIdx = -1;
      for(let b=0;b<batchSize;b++){
        const rng = mulberry32(Math.floor(rngBase()*1e9));
        const n = Math.floor(rng()*(countMax-countMin+1)) + countMin;
        const {data, imageData} = createImageBuffer(size, n, dotRadius, rng, allowOverlap);
        batchX.set(data, b*size*size);
        const classIdx = n - countMin;
        batchY[b*opts.numClasses + classIdx] = 1;
        // store first image of each batch for preview
        if(b===0 && !forValidation){
          firstImageData = imageData;
          firstImageTensor = data.slice(); // copy
          firstClassIdx = classIdx;
        }
      }
      const xs = tf.tensor4d(batchX, [batchSize, size, size, 1]);
      const ys = tf.tensor2d(batchY, [batchSize, opts.numClasses]);
      if(firstImageData){
        globalPreviewImages.push({imageData: firstImageData, tensorData: firstImageTensor, classIdx: firstClassIdx});
      }
      yield {xs, ys};
    }
  }
}

function buildModel(inputSize, numClasses){
  const m = tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[inputSize,inputSize,1], filters:16, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:64, activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.dense({units:numClasses, activation:'softmax'}));
  m.compile({optimizer:tf.train.adam(), loss:'categoricalCrossentropy', metrics:['accuracy']});
  return m;
}

function printModelArch(m){
  if(!m){ el('modelArch').textContent = '(not built)'; return; }
  const lines = [];
  lines.push(`${m.name} — layers: ${m.layers.length}`);
  m.layers.forEach((L,i)=>{
    const out = L.outputShape;
    lines.push(`${i}: ${L.getClassName()}  out=${JSON.stringify(out)}  params=${L.countParams()}`);
  });
  lines.push(`Total params: ${m.countParams()}`);
  el('modelArch').textContent = lines.join('\n');
}

function setupChart(){
  const ctx = el('accChart').getContext('2d');
  if(chart) chart.destroy();
  chart = new Chart(ctx, {
    type:'line', data:{labels:[], datasets:[{label:'Train Acc',borderColor:'blue',data:[]},{label:'Val Acc',borderColor:'green',data:[]} ]}, options:{animation:false,scales:{y:{min:0,max:1}}}
  });
}

async function startTrainingLoop(){
  // read options
  const opts = {
    overlapMode: el('overlapMode').value,
    countMin: parseInt(el('countMin').value,10),
    countMax: parseInt(el('countMax').value,10),
    batchSize: parseInt(el('batchSize').value,10),
    epochs: parseInt(el('epochs').value,10),
    updateEvery: parseInt(el('updateEvery').value,10),
    size: parseInt(el('imageSize').value,10),
    dotRadius: parseInt(el('dotRadius').value,10),
    seed: parseInt(el('seed').value,10),
    stepsPerEpoch: parseInt(el('stepsPerEpoch').value,10),
    valSteps: parseInt(el('valSteps').value,10),
  };
  opts.allowOverlap = opts.overlapMode === 'allow';
  opts.numClasses = opts.countMax - opts.countMin + 1;

  // build model
  if(model) model.dispose();
  model = buildModel(opts.size, opts.numClasses);
  printModelArch(model);

  setupChart();

  // datasets
  globalPreviewImages = [];
  const trainGen = makeGenerator(opts, false);
  const valGen = makeGenerator(opts, true);
  const trainDataset = tf.data.generator(trainGen).prefetch(2);
  const valDataset = tf.data.generator(valGen).prefetch(2);
  const valIterator = await valDataset.iterator();

  running = true; pauseRequested = false;
  el('progressLabel').textContent = 'Training';

  let totalBatches = 0;
  for(let e=0;e<opts.epochs && running; e++){
    el('progressLabel').textContent = `Epoch ${e+1}/${opts.epochs}`;
    // per-epoch fit for better pause control
    let batchCounter = 0;
    const lossAcc = [];
    await model.fitDataset(trainDataset, {
      epochs:1,
      batchesPerEpoch: opts.stepsPerEpoch,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          batchCounter++;
          totalBatches++;
          if(globalPreviewImages.length>0){
            const p = globalPreviewImages.shift();
            drawPreview(p.imageData);
            // run prediction on this image
            const inputTensor = tf.tensor4d(p.tensorData, [1, opts.size, opts.size, 1]);
            const pred = model.predict(inputTensor);
            const predClass = pred.argMax(-1).dataSync()[0];
            const actualDots = p.classIdx + opts.countMin;
            const predDots = predClass + opts.countMin;
            el('predLabel').textContent = `Predicted: ${predDots} dots | Actual: ${actualDots} dots`;
            inputTensor.dispose();
            pred.dispose();
          }
          if(logs.loss!=null) el('lossInfo').textContent = logs.loss.toFixed(4);
          if(logs.acc!=null) el('accInfo').textContent = logs.acc.toFixed(4);
          el('batchInfo').textContent = `${batchCounter} (total ${totalBatches})`;
          chart.data.labels.push(totalBatches);
          chart.data.datasets[0].data.push(logs.acc||0);
          
          // Calculate validation accuracy on one batch
          const valBatch = await valIterator.next();
          if(!valBatch.done){
            const valResult = await model.evaluate(valBatch.value.xs, valBatch.value.ys);
            const valAcc = await valResult[1].data();
            chart.data.datasets[1].data.push(valAcc[0]);
            valResult[0].dispose();
            valResult[1].dispose();
            valBatch.value.xs.dispose();
            valBatch.value.ys.dispose();
          } else {
            chart.data.datasets[1].data.push(null);
          }
          
          chart.update();
          if(pauseRequested){ model.stopTraining = true; }
          // Add delay to watch images being presented
          await tf.nextFrame();
          await new Promise(resolve => setTimeout(resolve, 100));
        },
        onEpochEnd: async (epochIdx, logs) => {
          await tf.nextFrame();
        }
      }
    });
    if(pauseRequested){ el('progressLabel').textContent = 'Paused'; break; }
  }
  running = false;
  if(!pauseRequested) el('progressLabel').textContent = 'Done';
}

function drawPreview(imageData){
  const c = el('previewCanvas');
  const ctx = c.getContext('2d');
  // scale imageData to canvas
  const tmp = document.createElement('canvas'); tmp.width = imageData.width; tmp.height = imageData.height;
  tmp.getContext('2d').putImageData(imageData,0,0);
  ctx.clearRect(0,0,c.width,c.height);
  ctx.drawImage(tmp,0,0,c.width,c.height);
}

function wireUI(){
  el('startBtn').addEventListener('click', ()=>{
    if(running) return; startTrainingLoop().catch(e=>{console.error(e); alert(e.message)});
  });
  el('pauseBtn').addEventListener('click', ()=>{
    if(!running) return; pauseRequested = true;
  });
  el('resetBtn').addEventListener('click', ()=>{
    pauseRequested = false; running = false; if(model) model.dispose(); model=null; printModelArch(null); setupChart(); el('progressLabel').textContent='Idle'; el('lossInfo').textContent='-'; el('accInfo').textContent='-'; el('batchInfo').textContent='-'; globalPreviewImages=[];
  });
}

window.addEventListener('load', ()=>{
  wireUI(); setupChart(); printModelArch(null);
});
