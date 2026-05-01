const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 10,
  title: 'DPT-style 4 层选择实证'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("DPT-style 4 层选择实证", {
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
  slide.addText("RESULT 5 · Layer ablation", {
    x: 7.0, y: 0.32, w: 2.8, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("1000 张真实图片：inter-output cosine + per-output magnitude balance", {
    x: 0.4, y: 0.95, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // Table on left
  const headerFill = { color: theme.primary };
  const headerText = { color: "FFFFFF", bold: true, fontSize: 11, fontFace: "Microsoft YaHei", align: "center", valign: "middle" };
  const cellOpt = { color: theme.primary, fontSize: 10, fontFace: "Arial", align: "center", valign: "middle" };
  const cellOptZh = { color: theme.primary, fontSize: 10, fontFace: "Microsoft YaHei", align: "center", valign: "middle" };
  const highlightFill = { color: theme.bg };
  const projectFill = { color: "e0f2fe" };

  const tableRows = [
    [
      { text: "candidate", options: { ...headerText, fill: headerFill } },
      { text: "layers (1-based)", options: { ...headerText, fill: headerFill } },
      { text: "mean cos", options: { ...headerText, fill: headerFill } },
      { text: "max/min mag", options: { ...headerText, fill: headerFill } },
      { text: "评价", options: { ...headerText, fill: headerFill } }
    ],
    [
      { text: "project", options: { ...cellOpt, fill: projectFill, bold: true, color: theme.secondary } },
      { text: "4/12/16/20", options: { ...cellOpt, fill: projectFill } },
      { text: "0.383", options: { ...cellOpt, fill: projectFill } },
      { text: "12.6× ★", options: { ...cellOpt, fill: projectFill, bold: true, color: theme.secondary } },
      { text: "最平衡（项目当前）", options: { ...cellOptZh, fill: projectFill, bold: true, color: theme.secondary } }
    ],
    [
      { text: "dpt", options: cellOpt },
      { text: "5/11/17/23", options: cellOpt },
      { text: "0.299 ★", options: { ...cellOpt, bold: true, color: theme.accent } },
      { text: "31.9×", options: cellOpt },
      { text: "最分散", options: cellOptZh }
    ],
    [
      { text: "late", options: cellOpt },
      { text: "6/12/18/24", options: cellOpt },
      { text: "0.339", options: cellOpt },
      { text: "84×", options: { ...cellOpt, bold: true, color: "dc2626" } },
      { text: "末层 magnitude 爆炸", options: { ...cellOptZh, color: "dc2626" } }
    ]
  ];

  slide.addTable(tableRows, {
    x: 0.4, y: 1.4, w: 5.0,
    colW: [0.7, 1.05, 0.85, 0.95, 1.45],
    rowH: 0.45,
    border: { type: "solid", pt: 0.5, color: theme.light }
  });

  // Right side: figure
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.5, y: 1.3, w: 4.2, h: 3.3,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.75 }
  });
  slide.addImage({
    path: "imgs/layer_ablation_diversity_vs_balance.svg",
    x: 5.55, y: 1.35, w: 4.1, h: 3.2
  });

  // Caption for figure
  slide.addText("Diversity (cos) vs Magnitude balance (max/min)", {
    x: 5.5, y: 4.65, w: 4.2, h: 0.3,
    fontSize: 10, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  // Bottom callout
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.55, w: 5.0, h: 0.6,
    fill: { color: theme.secondary },
    line: { type: "none" }
  });
  slide.addText("结论：项目 [4,12,16,20] 是 diversity-magnitude 折中", {
    x: 0.4, y: 4.55, w: 5.0, h: 0.3,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });
  slide.addText("不是 DPT 简单照搬", {
    x: 0.4, y: 4.85, w: 5.0, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.bg, bold: false,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("10", {
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
  pres.writeFile({ fileName: "slide-10-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
