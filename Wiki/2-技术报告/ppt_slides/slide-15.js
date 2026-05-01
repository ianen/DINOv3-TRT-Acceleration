const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'summary',
  index: 15,
  title: 'Conclusion'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Top accent stripe (decorative — distinguishes summary)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.18,
    fill: { color: theme.primary },
    line: { type: "none" }
  });

  // Big "Conclusion" title — larger than regular content slides
  slide.addText("Conclusion", {
    x: 0.4, y: 0.35, w: 9.2, h: 0.7,
    fontSize: 36, fontFace: "Arial",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title accent line — longer for summary feel
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 1.05, w: 2.4, h: 0,
    line: { color: theme.accent, width: 3 }
  });

  // Section label
  slide.addText("SUMMARY · 5 项核心结论", {
    x: 6.7, y: 0.5, w: 2.9, h: 0.4,
    fontSize: 13, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // 5 bullets with large checkmarks
  const conclusions = [
    "BF16 prefer 是 G2 ideal region 唯一候选 — 顶点 3.86× speedup + cos ≥ 0.998",
    "INT8 全路径 sensitivity 已闭合（5 paths × 12 points）",
    "V1.0+V1.1+V1.2 三层 mixed-precision 等价 negative",
    "跨语言 parity 三档 bit-identical — 部署可信度高",
    "V1.3 QAT 是 future work 唯一可能跨过 G2 的路径"
  ];

  const startY = 1.35;
  const rowH = 0.5;
  const rowGap = 0.08;

  conclusions.forEach((text, i) => {
    const y = startY + i * (rowH + rowGap);

    // Large primary checkmark circle
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: y + 0.05, w: 0.4, h: 0.4,
      fill: { color: theme.primary },
      line: { type: "none" }
    });
    slide.addText("v", {
      x: 0.5, y: y + 0.05, w: 0.4, h: 0.4,
      fontSize: 18, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });

    // Conclusion text — larger font for summary feel
    slide.addText(text, {
      x: 1.05, y: y, w: 8.5, h: rowH,
      fontSize: 15, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Bottom highlight — boxed in primary
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.55, w: 9.2, h: 0.55,
    fill: { color: theme.primary },
    line: { type: "none" }
  });
  slide.addText("BF16 prefer 是 RTX 5080 + TRT 10.13 + DINOv3 ViT-L 上唯一在 G2 ideal region 的候选", {
    x: 0.4, y: 4.55, w: 9.2, h: 0.55,
    fontSize: 14, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("15", {
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
  pres.writeFile({ fileName: "slide-15-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
