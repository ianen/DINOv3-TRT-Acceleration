const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 8,
  title: 'INT8 路径完整 sensitivity（12 点 tradeoff）'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title bar (full width primary color band for showcase emphasis)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.9,
    fill: { color: theme.primary },
    line: { type: "none" }
  });

  // Accent stripe under title bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.9, w: 10, h: 0.08,
    fill: { color: theme.accent },
    line: { type: "none" }
  });

  // Title
  slide.addText("INT8 路径完整 sensitivity", {
    x: 0.4, y: 0.1, w: 6.5, h: 0.4,
    fontSize: 24, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "left", valign: "middle"
  });

  // Subtitle in title bar
  slide.addText("12 点 tradeoff · BF16 唯一进入 G2 ideal region", {
    x: 0.4, y: 0.5, w: 6.5, h: 0.35,
    fontSize: 13, fontFace: "Microsoft YaHei",
    color: theme.bg, bold: false,
    align: "left", valign: "middle"
  });

  // Section label (right of title)
  slide.addText("RESULT 3 · SHOWCASE", {
    x: 7.0, y: 0.3, w: 2.8, h: 0.4,
    fontSize: 13, fontFace: "Arial",
    color: theme.accent, italic: true, bold: true,
    align: "right", valign: "middle"
  });

  // Main scatter chart (large, center)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 1.45, y: 1.05, w: 7.1, h: 4.0,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.75 }
  });
  slide.addImage({
    path: "imgs/benchmark_bf16_vs_int8_tradeoff.svg",
    x: 1.5, y: 1.05, w: 7.0, h: 4.0
  });

  // Top-right callout: BF16 prefer star
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.4, y: 1.25, w: 2.0, h: 0.65,
    fill: { color: theme.accent },
    line: { color: "FFFFFF", width: 1 }
  });
  slide.addText("★ BF16 prefer", {
    x: 6.4, y: 1.27, w: 2.0, h: 0.3,
    fontSize: 11, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });
  slide.addText("唯一进入 G2 ideal region", {
    x: 6.4, y: 1.55, w: 2.0, h: 0.3,
    fontSize: 9, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: false,
    align: "center", valign: "middle"
  });

  // Bottom callout: 12 points decomposition
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 5.15, w: 6.5, h: 0.4,
    fill: { color: theme.secondary },
    line: { type: "none" }
  });
  slide.addText("12 点 = 9 INT8 + FP8 default + FP8 partial layer19 + V1.2 ONNX-stripped", {
    x: 0.4, y: 5.15, w: 6.5, h: 0.4,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  // Bottom legend / axes
  slide.addText("X = feat_layer_20 cosine_mean | Y = trtexec b8 latency speedup vs FP32", {
    x: 0.4, y: 5.2, w: 9.2, h: 0.3,
    fontSize: 10, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "center", valign: "bottom"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("08", {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = { primary: "1e3a5f", secondary: "2563eb", accent: "0ea5e9", light: "94a3b8", bg: "f8fafc" };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-08-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
