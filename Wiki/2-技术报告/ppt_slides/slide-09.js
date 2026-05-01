const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 9,
  title: 'Mixed-Precision 三种工具链等价 negative'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("Mixed-Precision 三种工具链等价 negative", {
    x: 0.4, y: 0.25, w: 7, h: 0.55,
    fontSize: 26, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label (right of title)
  slide.addText("RESULT 4 · Convergence proof", {
    x: 7.0, y: 0.32, w: 2.8, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("V1.0 + V1.1 + V1.2 三层联合证据", {
    x: 0.4, y: 0.95, w: 9, h: 0.35,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // Table
  const headerFill = { color: theme.primary };
  const headerText = { color: "FFFFFF", bold: true, fontSize: 12, fontFace: "Microsoft YaHei", align: "center", valign: "middle" };
  const cellOpt = { color: theme.primary, fontSize: 11, fontFace: "Arial", align: "left", valign: "middle" };
  const cellOptCenter = { color: theme.primary, fontSize: 11, fontFace: "Arial", align: "center", valign: "middle" };
  const cellOptZh = { color: theme.primary, fontSize: 11, fontFace: "Microsoft YaHei", align: "left", valign: "middle" };
  const highlightFill = { color: theme.bg };
  const highlightOpt = { color: theme.secondary, bold: true, fontSize: 11, fontFace: "Arial", align: "left", valign: "middle" };
  const highlightOptZh = { color: theme.secondary, bold: true, fontSize: 11, fontFace: "Microsoft YaHei", align: "left", valign: "middle" };
  const highlightOptCenter = { color: theme.secondary, bold: true, fontSize: 11, fontFace: "Arial", align: "center", valign: "middle" };

  const tableRows = [
    [
      { text: "路径", options: { ...headerText, fill: headerFill } },
      { text: "工具链层", options: { ...headerText, fill: headerFill } },
      { text: "cos_min", options: { ...headerText, fill: headerFill } },
      { text: "b8 speedup", options: { ...headerText, fill: headerFill } }
    ],
    [
      { text: "Full SmoothQuant α=0.8 (V1.0 baseline)", options: cellOptZh },
      { text: "PyTorch ModelOpt", options: cellOpt },
      { text: "0.968", options: cellOptCenter },
      { text: "3.48×", options: cellOptCenter }
    ],
    [
      { text: "ModelOpt disable_quantizer skip 16-19", options: cellOptZh },
      { text: "PyTorch", options: cellOpt },
      { text: "0.971 (+0.003)", options: cellOptCenter },
      { text: "2.41× (-30%)", options: cellOptCenter }
    ],
    [
      { text: "trtexec --layerPrecisions=l16-19:fp32", options: cellOptZh },
      { text: "TRT command-line", options: cellOpt },
      { text: "0.9683 (≈)", options: cellOptCenter },
      { text: "3.43× (≈)", options: cellOptCenter }
    ],
    [
      { text: "V1.2 ONNX-level Q/DQ stripping", options: { ...highlightOptZh, fill: highlightFill } },
      { text: "ONNX library", options: { ...highlightOpt, fill: highlightFill } },
      { text: "0.9705", options: { ...highlightOptCenter, fill: highlightFill } },
      { text: "2.39×", options: { ...highlightOptCenter, fill: highlightFill } }
    ]
  ];

  slide.addTable(tableRows, {
    x: 0.4, y: 1.45, w: 9.2,
    colW: [3.5, 2.0, 1.6, 2.1],
    rowH: 0.45,
    border: { type: "solid", pt: 0.5, color: theme.light }
  });

  // Bullets - 3 conclusions
  const bullets = [
    "三种独立工具链 → cos_min 差 0.003 / speed 差 0.02× → 等价 negative",
    "Convergence proof：排除\"工具链 bug\"作为根本原因",
    "Root cause 在更上游 — 见 Page 11"
  ];

  const bulletStartY = 3.85;
  const bulletGap = 0.42;

  bullets.forEach((b, i) => {
    const y = bulletStartY + i * bulletGap;

    // Bullet marker (small filled square)
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: y + 0.13, w: 0.15, h: 0.15,
      fill: { color: theme.accent },
      line: { type: "none" }
    });

    // Bullet text
    slide.addText(b, {
      x: 0.75, y: y, w: 8.8, h: 0.4,
      fontSize: 13, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("09", {
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
  pres.writeFile({ fileName: "slide-09-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
