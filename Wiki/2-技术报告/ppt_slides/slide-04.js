const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 4,
  title: '架构决策（V1.0.1 主计划 ADR-001~009）'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("架构决策（V1.0.1 主计划 ADR-001~009）", {
    x: 0.4, y: 0.25, w: 7.5, h: 0.55,
    fontSize: 24, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label
  slide.addText("Method · Architecture", {
    x: 7.4, y: 0.32, w: 2.0, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // ADR rows (left column)
  const adrs = [
    { id: "ADR-001", text: "4 个 output binding feat_layer_{4,12,16,20}, [B,197,1024], 裁剪 register tokens" },
    { id: "ADR-002+009", text: "静态分辨率 + 动态 batch（每分辨率独立 engine）" },
    { id: "ADR-003", text: "INT8 主路径：ModelOpt 显式 Q/DQ" },
    { id: "ADR-007", text: "RoPE 处理：源码改造消除 ONNX If 节点" },
    { id: "ADR-008", text: "TRT 10.13+ 锁定（Blackwell sm_120 支持下限）" }
  ];

  const startY = 1.1;
  const rowH = 0.62;
  const rowGap = 0.05;

  adrs.forEach((adr, i) => {
    const y = startY + i * (rowH + rowGap);

    // Row background (subtle)
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: y, w: 5.6, h: rowH,
      fill: { color: "FFFFFF" },
      line: { color: theme.light, width: 0.4 }
    });

    // Left accent stripe
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: y, w: 0.08, h: rowH,
      fill: { color: theme.accent },
      line: { type: "none" }
    });

    // ADR ID (bold, secondary color)
    slide.addText(adr.id, {
      x: 0.55, y: y, w: 1.4, h: rowH,
      fontSize: 12, fontFace: "Arial",
      color: theme.secondary, bold: true,
      align: "left", valign: "middle"
    });

    // Description
    slide.addText(adr.text, {
      x: 1.95, y: y, w: 4.0, h: rowH,
      fontSize: 11.5, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Right side: visual flow diagram
  // Title for the visual
  slide.addText("Feature Extraction Flow", {
    x: 6.3, y: 1.1, w: 3.4, h: 0.35,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, bold: true,
    align: "center", valign: "middle",
    charSpacing: 4
  });

  // ViT-L block (top)
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.5, y: 1.55, w: 3.0, h: 0.6,
    fill: { color: theme.primary },
    line: { type: "none" },
    rectRadius: 0.08
  });
  slide.addText("ViT-L/16 (24 blocks)", {
    x: 6.5, y: 1.55, w: 3.0, h: 0.6,
    fontSize: 14, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  // Down arrow (line + arrowhead approximation)
  slide.addShape(pres.shapes.LINE, {
    x: 8.0, y: 2.18, w: 0, h: 0.35,
    line: { color: theme.secondary, width: 2, endArrowType: "triangle" }
  });

  // 4 hooks block
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.5, y: 2.6, w: 3.0, h: 0.55,
    fill: { color: theme.secondary },
    line: { type: "none" },
    rectRadius: 0.08
  });
  slide.addText("4 hooks @ layer 4/12/16/20", {
    x: 6.5, y: 2.6, w: 3.0, h: 0.55,
    fontSize: 13, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  // Down arrow
  slide.addShape(pres.shapes.LINE, {
    x: 8.0, y: 3.18, w: 0, h: 0.35,
    line: { color: theme.secondary, width: 2, endArrowType: "triangle" }
  });

  // 4 outputs row (4 small boxes)
  const outNames = ["L4", "L12", "L16", "L20"];
  const outStartX = 6.5;
  const outBoxW = 0.68;
  const outBoxGap = 0.1;

  outNames.forEach((name, i) => {
    const ox = outStartX + i * (outBoxW + outBoxGap);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: ox, y: 3.6, w: outBoxW, h: 0.55,
      fill: { color: theme.accent },
      line: { type: "none" },
      rectRadius: 0.06
    });
    slide.addText(name, {
      x: ox, y: 3.6, w: outBoxW, h: 0.55,
      fontSize: 13, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });
  });

  // Caption under outputs
  slide.addText("[B, 197, 1024] × 4 outputs", {
    x: 6.3, y: 4.25, w: 3.4, h: 0.35,
    fontSize: 11, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("04", {
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
  pres.writeFile({ fileName: "slide-04-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
