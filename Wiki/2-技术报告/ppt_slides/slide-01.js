const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'cover',
  index: 1,
  title: 'DINOv3 ViT-L/16 多尺度 4 输出 TensorRT 加速研究'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Top accent stripe (decorative)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.35,
    fill: { color: theme.primary },
    line: { type: "none" }
  });

  // Bottom accent stripe (decorative, thinner)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.475, w: 10, h: 0.15,
    fill: { color: theme.accent },
    line: { type: "none" }
  });

  // Small caption above title
  slide.addText("PolyU 研究型项目 · 视觉基础模型推理加速", {
    x: 0.5, y: 1.15, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Microsoft YaHei",
    color: theme.light, bold: false,
    align: "center", valign: "middle",
    charSpacing: 4
  });

  // Main title (large, primary)
  slide.addText("DINOv3 ViT-L/16 多尺度 4 输出", {
    x: 0.5, y: 1.65, w: 9, h: 0.85,
    fontSize: 38, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "center", valign: "middle"
  });

  slide.addText("TensorRT 加速研究", {
    x: 0.5, y: 2.5, w: 9, h: 0.85,
    fontSize: 38, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "center", valign: "middle"
  });

  // Horizontal accent line under title
  slide.addShape(pres.shapes.LINE, {
    x: 3.5, y: 3.55, w: 3, h: 0,
    line: { color: theme.accent, width: 2.25 }
  });

  // Subtitle
  slide.addText("Blackwell sm_120 + TRT 10.13 + INT8 sensitivity 完整 ablation", {
    x: 0.5, y: 3.7, w: 9, h: 0.5,
    fontSize: 20, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "center", valign: "middle"
  });

  // Author / affiliation block
  slide.addText("PolyU · 2026-05", {
    x: 0.5, y: 4.7, w: 9, h: 0.4,
    fontSize: 16, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = { primary: "1e3a5f", secondary: "2563eb", accent: "0ea5e9", light: "94a3b8", bg: "f8fafc" };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-01-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
