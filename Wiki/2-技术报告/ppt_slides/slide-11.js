const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 11,
  title: '为什么 INT8 cos_min < 0.99？Root Cause 分析'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("为什么 INT8 cos_min < 0.99？Root Cause 分析", {
    x: 0.4, y: 0.25, w: 8, h: 0.55,
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
  slide.addText("DISCUSSION · Root Cause", {
    x: 7.0, y: 0.32, w: 2.8, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Top arrow + label: noise accumulation
  slide.addText("INT8 噪声累积 ~10⁻²", {
    x: 0.4, y: 1.0, w: 9.2, h: 0.3,
    fontSize: 13, fontFace: "Arial",
    color: theme.accent, bold: true,
    align: "center", valign: "middle"
  });

  // Arrow line under accumulation label
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 1.4, w: 9.0, h: 0,
    line: { color: theme.accent, width: 2, endArrowType: "triangle" }
  });

  // Block diagram: 24 transformer blocks B0..B23
  const blockStartX = 0.5;
  const blockY = 1.65;
  const blockW = 0.36;
  const blockH = 0.45;
  const blockGap = 0.025;
  const totalBlocks = 24;

  for (let i = 0; i < totalBlocks; i++) {
    const x = blockStartX + i * (blockW + blockGap);
    let fillColor;
    // Blocks 0-15: INT8 PTQ (red shade)
    // Blocks 16-19: FP32 fallback (green shade)
    // Blocks 20-23: INT8 PTQ (red shade)
    if (i >= 16 && i <= 19) {
      fillColor = "10b981"; // green
    } else {
      fillColor = "ef4444"; // red
    }

    // Block rect
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: blockY, w: blockW, h: blockH,
      fill: { color: fillColor },
      line: { color: "FFFFFF", width: 0.5 }
    });
    slide.addText(`B${i}`, {
      x: x, y: blockY, w: blockW, h: blockH,
      fontSize: 8, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });
  }

  // Vertical dashed line at block 16 boundary (between B15 and B16)
  const block16BoundaryX = blockStartX + 16 * (blockW + blockGap) - blockGap / 2;
  slide.addShape(pres.shapes.LINE, {
    x: block16BoundaryX, y: 1.5, w: 0, h: 0.85,
    line: { color: theme.primary, width: 1.25, dashType: "dash" }
  });
  // Vertical dashed line at block 20 boundary (between B19 and B20)
  const block20BoundaryX = blockStartX + 20 * (blockW + blockGap) - blockGap / 2;
  slide.addShape(pres.shapes.LINE, {
    x: block20BoundaryX, y: 1.5, w: 0, h: 0.85,
    line: { color: theme.primary, width: 1.25, dashType: "dash" }
  });

  // Below blocks: shaded annotation bands
  // Blocks 0-15: INT8 PTQ red
  const band0_15_w = 16 * (blockW + blockGap) - blockGap;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: blockStartX, y: 2.2, w: band0_15_w, h: 0.35,
    fill: { color: "fee2e2" },
    line: { color: "ef4444", width: 0.75 }
  });
  slide.addText("INT8 PTQ", {
    x: blockStartX, y: 2.2, w: band0_15_w, h: 0.35,
    fontSize: 10, fontFace: "Arial",
    color: "991b1b", bold: true,
    align: "center", valign: "middle"
  });

  // Blocks 16-19: FP32 fallback green
  const band16_19_x = blockStartX + 16 * (blockW + blockGap);
  const band16_19_w = 4 * (blockW + blockGap) - blockGap;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: band16_19_x, y: 2.2, w: band16_19_w, h: 0.35,
    fill: { color: "d1fae5" },
    line: { color: "10b981", width: 0.75 }
  });
  slide.addText("FP32 fallback (V1.2)", {
    x: band16_19_x, y: 2.2, w: band16_19_w, h: 0.35,
    fontSize: 8, fontFace: "Arial",
    color: "065f46", bold: true,
    align: "center", valign: "middle"
  });

  // Blocks 20-23: INT8 PTQ red
  const band20_23_x = blockStartX + 20 * (blockW + blockGap);
  const band20_23_w = 4 * (blockW + blockGap) - blockGap;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: band20_23_x, y: 2.2, w: band20_23_w, h: 0.35,
    fill: { color: "fee2e2" },
    line: { color: "ef4444", width: 0.75 }
  });
  slide.addText("INT8", {
    x: band20_23_x, y: 2.2, w: band20_23_w, h: 0.35,
    fontSize: 9, fontFace: "Arial",
    color: "991b1b", bold: true,
    align: "center", valign: "middle"
  });

  // Bullets - root cause analysis
  const bullets = [
    "Blocks 0-15 累积 INT8 量化误差 → 偏离 FP32 baseline ~10⁻² 量级",
    "到达 block 16 输入时已偏离，layer 16-19 即使 FP32 也无法 recover",
    "工具链 layer 不重要：PyTorch / TRT / ONNX 都 hit same TRT fallback",
    "唯一出路：从前段开始减少量化（V1.3 QAT）"
  ];

  const bulletStartY = 2.85;
  const bulletGap = 0.5;

  bullets.forEach((b, i) => {
    const y = bulletStartY + i * bulletGap;
    const isLast = i === bullets.length - 1;

    // Bullet marker
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: y + 0.12, w: 0.18, h: 0.18,
      fill: { color: isLast ? theme.accent : theme.secondary },
      line: { type: "none" }
    });

    // Bullet text - last one bold
    slide.addText(b, {
      x: 0.78, y: y, w: 8.8, h: 0.42,
      fontSize: 13, fontFace: "Microsoft YaHei",
      color: isLast ? theme.accent : theme.primary,
      bold: isLast,
      align: "left", valign: "middle"
    });
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("11", {
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
  pres.writeFile({ fileName: "slide-11-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
