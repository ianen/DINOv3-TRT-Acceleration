const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 14,
  title: 'V1.3 — Quantization-Aware Fine-Tuning（ADR-011 Proposed）'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("V1.3 — Quantization-Aware Fine-Tuning", {
    x: 0.4, y: 0.22, w: 7.5, h: 0.5,
    fontSize: 24, fontFace: "Arial",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.78, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label
  slide.addText("FUTURE WORK · ADR-011 Proposed", {
    x: 7.0, y: 0.28, w: 2.6, h: 0.4,
    fontSize: 11, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Top section subtitle
  slide.addText("路径设计（4 步）", {
    x: 0.4, y: 0.88, w: 6, h: 0.32,
    fontSize: 15, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: true,
    align: "left", valign: "middle"
  });

  // Top 4 bullets - path
  const pathBullets = [
    "从 SmoothQuant α=0.8 PTQ initialization 出发",
    "ImageNet val 50K + ModelOpt QAT mode + 1-5 epoch fine-tune",
    "期望 cos_min ≥ 0.99 同时保留 ≥ 3.0× speedup",
    "→ 首个完整满足 G2 ideal region 的 INT8 候选"
  ];

  const topStartY = 1.25;
  const topRowH = 0.36;
  const topRowGap = 0.04;

  pathBullets.forEach((text, i) => {
    const y = topStartY + i * (topRowH + topRowGap);

    // Numbered circle
    slide.addShape(pres.shapes.OVAL, {
      x: 0.45, y: y + 0.04, w: 0.28, h: 0.28,
      fill: { color: theme.accent },
      line: { type: "none" }
    });
    slide.addText(String(i + 1), {
      x: 0.45, y: y + 0.04, w: 0.28, h: 0.28,
      fontSize: 11, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });

    // Text
    slide.addText(text, {
      x: 0.85, y: y, w: 8.7, h: topRowH,
      fontSize: 13, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Divider
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 2.95, w: 9.2, h: 0,
    line: { color: theme.light, width: 0.5 }
  });

  // Bottom section title
  slide.addText("4 条启动门槛", {
    x: 0.4, y: 3.05, w: 6, h: 0.35,
    fontSize: 15, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: true,
    align: "left", valign: "middle"
  });

  // Status badge - all blocked
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.6, y: 3.07, w: 2.0, h: 0.32,
    fill: { color: theme.primary },
    line: { type: "none" },
    rectRadius: 0.04
  });
  slide.addText("ALL BLOCKED", {
    x: 7.6, y: 3.07, w: 2.0, h: 0.32,
    fontSize: 11, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  // Bottom 4 status items
  const gates = [
    "ImageNet val 50K unblock",
    "训练资源 ≥ 5 GPU-day",
    "时间预算 1-2 个月（含论文写作）",
    "下游任务 baseline（depth/segmentation FP32 + DPT 头）"
  ];

  const gateStartY = 3.5;
  const gateRowH = 0.3;
  const gateRowGap = 0.04;

  gates.forEach((text, i) => {
    const y = gateStartY + i * (gateRowH + gateRowGap);

    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: y, w: 9.2, h: gateRowH,
      fill: { color: "FFFFFF" },
      line: { color: theme.light, width: 0.3 }
    });

    // X mark cross
    slide.addText("X", {
      x: 0.5, y: y, w: 0.4, h: gateRowH,
      fontSize: 14, fontFace: "Arial",
      color: theme.secondary, bold: true,
      align: "center", valign: "middle"
    });

    // Text
    slide.addText(text, {
      x: 1.0, y: y, w: 8.5, h: gateRowH,
      fontSize: 12, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Note at bottom
  slide.addText("4 条门槛全未满足 — 暂不启动，是 V1.3 / 论文阶段任务", {
    x: 0.4, y: 4.95, w: 8.5, h: 0.32,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "left", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("14", {
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
  pres.writeFile({ fileName: "slide-14-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
