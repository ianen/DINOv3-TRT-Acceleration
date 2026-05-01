const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 7,
  title: 'BF16 prefer 精度结果'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("BF16 prefer 精度结果", {
    x: 0.4, y: 0.25, w: 7, h: 0.55,
    fontSize: 28, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label (right of title)
  slide.addText("RESULT 2 · BF16 cosine", {
    x: 7.0, y: 0.32, w: 2.3, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("Imagenette 1000 张 vs FP32", {
    x: 0.4, y: 0.95, w: 6, h: 0.35,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // Cosine table - left side
  const headerFill = { color: theme.primary };
  const headerText = { color: "FFFFFF", bold: true, fontSize: 11, fontFace: "Arial", align: "center", valign: "middle" };
  const cellText = { color: theme.primary, fontSize: 11, fontFace: "Arial", align: "center", valign: "middle" };
  const cellTextBold = { color: theme.secondary, fontSize: 11, fontFace: "Arial", align: "center", valign: "middle", bold: true };
  const highlightFill = { color: theme.accent };
  const highlightText = { color: "FFFFFF", bold: true, fontSize: 11, fontFace: "Arial", align: "center", valign: "middle" };

  const tableRows = [
    [
      { text: "Resolution", options: { ...headerText, fill: headerFill } },
      { text: "feat_layer_4", options: { ...headerText, fill: headerFill } },
      { text: "feat_layer_12", options: { ...headerText, fill: headerFill } },
      { text: "feat_layer_16", options: { ...headerText, fill: headerFill } },
      { text: "feat_layer_20", options: { ...headerText, fill: headerFill } }
    ],
    [
      { text: "r224", options: { ...cellTextBold } },
      { text: "0.999933", options: cellText },
      { text: "0.999664", options: cellText },
      { text: "0.998943", options: cellText },
      { text: "0.998749", options: cellText }
    ],
    [
      { text: "r336", options: { ...cellTextBold } },
      { text: "0.999891", options: cellText },
      { text: "0.999276", options: cellText },
      { text: "0.998394", options: cellText },
      { text: "0.998493", options: cellText }
    ],
    [
      { text: "r518", options: { ...cellTextBold } },
      { text: "0.999868", options: cellText },
      { text: "0.999075", options: cellText },
      { text: "0.998604", options: cellText },
      { text: "0.999171 ★", options: { ...highlightText, fill: highlightFill } }
    ]
  ];

  slide.addTable(tableRows, {
    x: 0.4, y: 1.4, w: 5.0,
    colW: [0.9, 1.025, 1.025, 1.025, 1.025],
    rowH: 0.4,
    border: { type: "solid", pt: 0.5, color: theme.light }
  });

  // Annotation below table - threshold
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.5, w: 5.0, h: 0.5,
    fill: { color: "FFFFFF" },
    line: { color: theme.secondary, width: 0.75 }
  });
  slide.addText("全部 ≥ G1 阈值（cosine ≥ 0.99，实际 ≥ 0.998）", {
    x: 0.4, y: 3.5, w: 5.0, h: 0.5,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: true,
    align: "center", valign: "middle"
  });

  // Annotation below table - r518 callout
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.1, w: 5.0, h: 0.85,
    fill: { color: theme.bg },
    line: { color: theme.accent, width: 1 }
  });
  slide.addText("★ r518 patch token 多 → BF16 误差被稀释", {
    x: 0.5, y: 4.15, w: 4.8, h: 0.3,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });
  slide.addText("反直觉：r518 feat_layer_20 cos 最高（0.999171）", {
    x: 0.5, y: 4.45, w: 4.8, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "left", valign: "middle"
  });
  slide.addText("更长序列 → softmax 平滑 → 量化噪声平均化", {
    x: 0.5, y: 4.7, w: 4.8, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "left", valign: "middle"
  });

  // Right side: cosine_min figure
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.5, y: 2.5, w: 4.0, h: 2.5,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.75 }
  });
  slide.addImage({
    path: "imgs/benchmark_bf16_cosine_min.svg",
    x: 5.55, y: 2.55, w: 3.9, h: 2.4
  });

  // Caption for figure
  slide.addText("BF16 cosine min · 三档分辨率 × 4 输出", {
    x: 5.5, y: 1.9, w: 4.0, h: 0.5,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("07", {
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
  pres.writeFile({ fileName: "slide-07-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
