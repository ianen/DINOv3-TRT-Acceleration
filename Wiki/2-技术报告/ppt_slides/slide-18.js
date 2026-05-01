const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 18,
  title: 'Backup Slides（应对追问）'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("Backup Slides（应对追问）", {
    x: 0.4, y: 0.25, w: 7.5, h: 0.55,
    fontSize: 28, fontFace: "Microsoft YaHei",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label
  slide.addText("APPENDIX · Q1-Q10 之外", {
    x: 7.0, y: 0.32, w: 2.6, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("6 项备用资料 — 涵盖深入追问场景", {
    x: 0.4, y: 0.95, w: 8, h: 0.35,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // 6 bullets in 2x3 grid
  const backups = [
    { tag: "B1", text: "ADR-007 RoPE 改造前后对比" },
    { tag: "B2", text: "SmoothQuant α-sweep 详细数据（α=0.5/0.7/0.8）" },
    { tag: "B3", text: "4 层 ablation 详细 magnitude 表（per-layer L2 norm）" },
    { tag: "B4", text: "C++ runtime parity 详细对照（max_abs / RMSE / cosine 全 4 输出）" },
    { tag: "B5", text: "V1.2 ONNX strip plan 示意（96 节点删 + 48 input slots rewire）" },
    { tag: "B6", text: "ADR-011 QAT 4 条启动门槛全状态表" }
  ];

  const startY = 1.45;
  const colW = 4.55;
  const colGap = 0.1;
  const rowH = 1.05;
  const rowGap = 0.12;

  backups.forEach((bk, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * (colW + colGap);
    const y = startY + row * (rowH + rowGap);

    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: colW, h: rowH,
      fill: { color: "FFFFFF" },
      line: { color: theme.light, width: 0.4 }
    });

    // Left accent stripe
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 0.08, h: rowH,
      fill: { color: theme.accent },
      line: { type: "none" }
    });

    // Tag (B1, B2, ...)
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x + 0.2, y: y + 0.18, w: 0.55, h: 0.34,
      fill: { color: theme.primary },
      line: { type: "none" },
      rectRadius: 0.05
    });
    slide.addText(bk.tag, {
      x: x + 0.2, y: y + 0.18, w: 0.55, h: 0.34,
      fontSize: 12, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });

    // Text
    slide.addText(bk.text, {
      x: x + 0.2, y: y + 0.55, w: colW - 0.3, h: 0.45,
      fontSize: 12, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Bottom note
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 5.0, w: 8.7, h: 0.4,
    fill: { color: theme.bg },
    line: { color: theme.secondary, width: 0.5 }
  });
  slide.addText("答辩官提问超出 Q1-Q10 时使用速答模板 — Wiki/2-技术报告/答辩问答预案_V1.0.0.md", {
    x: 0.4, y: 5.0, w: 8.7, h: 0.4,
    fontSize: 10.5, fontFace: "Microsoft YaHei",
    color: theme.secondary, italic: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("18", {
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
  pres.writeFile({ fileName: "slide-18-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
