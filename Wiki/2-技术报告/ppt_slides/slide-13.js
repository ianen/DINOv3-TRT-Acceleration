const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 13,
  title: 'Limitations'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("Limitations", {
    x: 0.4, y: 0.25, w: 7.5, h: 0.55,
    fontSize: 30, fontFace: "Arial",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label (right of title)
  slide.addText("DISCUSSION · 局限性", {
    x: 7.4, y: 0.32, w: 2.2, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("4 项已知约束（坦诚披露）", {
    x: 0.4, y: 0.95, w: 8, h: 0.35,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // 4 cards stacked vertically
  const limitations = [
    { letter: "D", text: "ImageNet val 50K gated 403 → Imagenette2-320（10 类 13K 张）替代" },
    { letter: "H", text: "TRT 10.13 + Blackwell BF16 + Q/DQ Myelin Fill 不兼容 → 强制 FP32 fallback" },
    { letter: "G", text: "单一硬件（RTX 5080 sm_120）→ Ada Lovelace / Hopper 行为可能不同" },
    { letter: "Q", text: "QAT 未实施（ADR-011 § 8 4 条启动门槛全未满足）" }
  ];

  const cardStartY = 1.45;
  const cardH = 0.78;
  const cardGap = 0.12;

  limitations.forEach((lim, i) => {
    const y = cardStartY + i * (cardH + cardGap);

    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: y, w: 9.2, h: cardH,
      fill: { color: "FFFFFF" },
      line: { color: theme.light, width: 0.4 }
    });

    // Left red-tinged accent stripe (using theme.secondary as red-tinged indicator)
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: y, w: 0.1, h: cardH,
      fill: { color: theme.secondary },
      line: { type: "none" }
    });

    // Icon-like OVAL with letter
    slide.addShape(pres.shapes.OVAL, {
      x: 0.7, y: y + 0.16, w: 0.46, h: 0.46,
      fill: { color: theme.primary },
      line: { type: "none" }
    });
    slide.addText(lim.letter, {
      x: 0.7, y: y + 0.16, w: 0.46, h: 0.46,
      fontSize: 14, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });

    // Description text
    slide.addText(lim.text, {
      x: 1.3, y: y, w: 8.2, h: cardH,
      fontSize: 14, fontFace: "Microsoft YaHei",
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
  slide.addText("13", {
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
  pres.writeFile({ fileName: "slide-13-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
