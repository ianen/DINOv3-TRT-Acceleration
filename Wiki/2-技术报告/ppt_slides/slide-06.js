const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 6,
  title: 'BF16 prefer 速度结果'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("BF16 prefer 速度结果", {
    x: 0.4, y: 0.25, w: 7.5, h: 0.55,
    fontSize: 26, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label
  slide.addText("Result · BF16 Speed", {
    x: 7.5, y: 0.32, w: 1.9, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle (smaller)
  slide.addText("locked 2752 MHz + spin-wait, vs FP32 baseline", {
    x: 0.4, y: 0.9, w: 9.0, h: 0.35,
    fontSize: 13, fontFace: "Microsoft YaHei",
    color: theme.secondary, italic: true,
    align: "left", valign: "middle"
  });

  // Speedup table (left side)
  const tableRows = [
    [
      { text: "Resolution", options: { fontFace: "Arial", bold: true, color: "FFFFFF", fill: { color: theme.primary }, align: "center", valign: "middle", fontSize: 12 } },
      { text: "b1",         options: { fontFace: "Arial", bold: true, color: "FFFFFF", fill: { color: theme.primary }, align: "center", valign: "middle", fontSize: 12 } },
      { text: "b8",         options: { fontFace: "Arial", bold: true, color: "FFFFFF", fill: { color: theme.primary }, align: "center", valign: "middle", fontSize: 12 } },
      { text: "b32",        options: { fontFace: "Arial", bold: true, color: "FFFFFF", fill: { color: theme.primary }, align: "center", valign: "middle", fontSize: 12 } }
    ],
    [
      { text: "r224",   options: { fontFace: "Arial", bold: true, color: theme.primary,   align: "center", valign: "middle", fontSize: 12 } },
      { text: "2.45×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12 } },
      { text: "2.81×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12 } },
      { text: "3.25×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12 } }
    ],
    [
      { text: "r336",   options: { fontFace: "Arial", bold: true, color: theme.primary,   align: "center", valign: "middle", fontSize: 12, fill: { color: "FFFFFF" } } },
      { text: "2.80×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12, fill: { color: "FFFFFF" } } },
      { text: "2.96×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12, fill: { color: "FFFFFF" } } },
      { text: "3.25×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12, fill: { color: "FFFFFF" } } }
    ],
    [
      { text: "r518",   options: { fontFace: "Arial", bold: true, color: theme.primary,   align: "center", valign: "middle", fontSize: 12 } },
      { text: "3.12×",  options: { fontFace: "Arial", color: theme.primary,               align: "center", valign: "middle", fontSize: 12 } },
      { text: "3.86× ★", options: { fontFace: "Arial", bold: true, color: "FFFFFF",       fill: { color: theme.accent }, align: "center", valign: "middle", fontSize: 13 } },
      { text: "—",      options: { fontFace: "Arial", color: theme.light,                 align: "center", valign: "middle", fontSize: 12 } }
    ]
  ];

  slide.addTable(tableRows, {
    x: 0.4, y: 1.4, w: 4.8,
    rowH: 0.5,
    colW: [1.4, 1.13, 1.13, 1.14],
    border: { type: "solid", color: theme.light, pt: 0.5 },
    fontFace: "Arial"
  });

  // Annotation under table
  slide.addText("★ 顶点 speedup（r518 b8）", {
    x: 0.4, y: 3.5, w: 4.8, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.accent, bold: true,
    align: "left", valign: "middle"
  });

  // C++ end-to-end note (highlighted strip)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.95, w: 4.8, h: 0.6,
    fill: { color: theme.primary },
    line: { type: "none" }
  });
  // Left accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.95, w: 0.1, h: 0.6,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("C++ end-to-end r518 b8: 3.40×", {
    x: 0.6, y: 3.95, w: 4.5, h: 0.3,
    fontSize: 13, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "left", valign: "middle"
  });
  slide.addText("（含 H2D + enqueue + D2H）", {
    x: 0.6, y: 4.22, w: 4.5, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "left", valign: "middle"
  });

  // Right side: SVG embed for BF16 speedup chart
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.45, y: 1.95, w: 4.1, h: 3.05,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.5 }
  });
  slide.addImage({
    path: "imgs/benchmark_trtexec_bf16_speedup.svg",
    x: 5.5, y: 2.0, w: 4.0, h: 3.0
  });
  slide.addText("trtexec BF16 vs FP32 GPU median latency speedup", {
    x: 5.45, y: 5.05, w: 4.1, h: 0.3,
    fontSize: 10, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("06", {
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
  pres.writeFile({ fileName: "slide-06-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
