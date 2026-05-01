// Compile all 18 slide modules into the final DINOv3-TRT-Acceleration PPTX.
// Run from this directory:  node compile.js

const path = require('path');
const pptxgen = require('pptxgenjs');

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'PolyU DINOv3-TRT-Acceleration';
pres.company = 'PolyU';
pres.title = 'DINOv3 ViT-L/16 多尺度 4 输出 TensorRT 加速研究';
pres.subject = 'V1.0+V1.1+V1.2 闭合证据链 + V1.3 QAT future work';

// Theme — must match slide modules exactly. Five keys only.
const theme = {
  primary: '1e3a5f',    // deep navy — titles, dark backgrounds
  secondary: '2563eb',  // vivid blue — body text accents
  accent: '0ea5e9',     // sky blue — highlights
  light: '94a3b8',      // slate gray — light accents
  bg: 'f8fafc',         // very light slate — background
};

const TOTAL_SLIDES = 18;

console.log(`[compile] generating ${TOTAL_SLIDES} slides ...`);
for (let i = 1; i <= TOTAL_SLIDES; i++) {
  const num = String(i).padStart(2, '0');
  const modulePath = path.join(__dirname, `slide-${num}.js`);
  // Clear from require cache for clean re-runs.
  delete require.cache[require.resolve(modulePath)];
  const mod = require(modulePath);
  if (typeof mod.createSlide !== 'function') {
    throw new Error(`slide-${num}.js does not export createSlide(pres, theme)`);
  }
  mod.createSlide(pres, theme);
  console.log(`  slide-${num} ✓`);
}

const outputPath = path.join(__dirname, 'output', 'DINOv3-TRT-Acceleration_V1.0.0.pptx');
pres.writeFile({ fileName: outputPath }).then((written) => {
  console.log(`[compile] wrote ${written}`);
});
