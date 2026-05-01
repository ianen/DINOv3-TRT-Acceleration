const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 3,
  title: '为什么要在消费级硬件上加速 ViT-L/16？'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("为什么要在消费级硬件上加速 ViT-L/16？", {
    x: 0.4, y: 0.25, w: 9.2, h: 0.55,
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
  slide.addText("Motivation", {
    x: 7.5, y: 0.32, w: 1.9, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Left column bullets (each with arrow / chevron mark)
  const bullets = [
    "DINOv3 ViT-L/16 LVD-1689M — Meta 最新视觉自监督基础模型",
    "4 输出多尺度特征（layer 4/12/16/20）→ DPT-style 融合",
    "ViT-L FP32 在 RTX 5080 batch 8 r224 = 28 ms — 部署成本高",
    "工程目标：BF16/INT8 加速到 ≥ 2× 同时保持 cosine ≥ 0.99"
  ];

  const startY = 1.15;
  const gap = 0.85;

  bullets.forEach((text, i) => {
    const y = startY + i * gap;

    // Arrow chevron (triangle)
    slide.addShape(pres.shapes.RIGHT_TRIANGLE, {
      x: 0.4, y: y + 0.12, w: 0.32, h: 0.32,
      fill: { color: theme.secondary },
      line: { type: "none" },
      rotate: 0
    });

    // Bullet text
    slide.addText(text, {
      x: 0.85, y: y, w: 5.0, h: 0.65,
      fontSize: 15, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });

    // Thin separator line under each bullet (except last)
    if (i < bullets.length - 1) {
      slide.addShape(pres.shapes.LINE, {
        x: 0.85, y: y + 0.7, w: 4.85, h: 0,
        line: { color: theme.light, width: 0.5, dashType: "dash" }
      });
    }
  });

  // Right-side callout box highlighting the core question
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.2, y: 1.5, w: 3.5, h: 2.6,
    fill: { color: theme.primary },
    line: { type: "none" }
  });

  // Callout corner accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.2, y: 1.5, w: 3.5, h: 0.2,
    fill: { color: theme.accent },
    line: { type: "none" }
  });

  // Callout label (English)
  slide.addText("CORE OBJECTIVE", {
    x: 6.35, y: 1.85, w: 3.2, h: 0.35,
    fontSize: 11, fontFace: "Arial",
    color: theme.accent, bold: true,
    align: "left", valign: "middle",
    charSpacing: 6
  });

  // Callout title
  slide.addText("核心问题", {
    x: 6.35, y: 2.2, w: 3.2, h: 0.55,
    fontSize: 22, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "left", valign: "middle"
  });

  // Inner divider in callout
  slide.addShape(pres.shapes.LINE, {
    x: 6.35, y: 2.85, w: 1.0, h: 0,
    line: { color: theme.accent, width: 2 }
  });

  // Callout body
  slide.addText("加速 ≥ 2×  ∧  cosine ≥ 0.99", {
    x: 6.35, y: 3.0, w: 3.2, h: 0.5,
    fontSize: 17, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "left", valign: "middle"
  });

  // Sub-text in callout
  slide.addText("延迟与精度的双约束最优化", {
    x: 6.35, y: 3.5, w: 3.2, h: 0.5,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "left", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("03", {
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
  pres.writeFile({ fileName: "slide-03-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
