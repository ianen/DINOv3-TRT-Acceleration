const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'section_divider',
  index: 17,
  title: 'Q&A'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();

  // INVERSE color scheme — full slide primary background
  slide.background = { color: theme.primary };

  // Top decorative accent stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.18,
    fill: { color: theme.accent },
    line: { type: "none" }
  });

  // Bottom decorative accent stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.45, w: 10, h: 0.18,
    fill: { color: theme.accent },
    line: { type: "none" }
  });

  // Decorative large OVAL (left side, soft)
  slide.addShape(pres.shapes.OVAL, {
    x: -1.2, y: 1.0, w: 3.0, h: 3.0,
    fill: { color: theme.secondary },
    line: { type: "none" }
  });

  // Decorative large OVAL (right side, soft)
  slide.addShape(pres.shapes.OVAL, {
    x: 8.2, y: 1.5, w: 3.0, h: 3.0,
    fill: { color: theme.accent },
    line: { type: "none" }
  });

  // Section pre-label (small, top center)
  slide.addText("SECTION  ·  17 / 18", {
    x: 0.5, y: 0.7, w: 9, h: 0.3,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, bold: false,
    align: "center", valign: "middle",
    charSpacing: 8
  });

  // Large centered "Questions?" — using bg color text on primary background
  slide.addText("Questions?", {
    x: 0.5, y: 1.7, w: 9, h: 1.5,
    fontSize: 72, fontFace: "Arial",
    color: theme.bg, bold: true,
    align: "center", valign: "middle"
  });

  // Decorative horizontal line accent
  slide.addShape(pres.shapes.LINE, {
    x: 4.0, y: 3.35, w: 2, h: 0,
    line: { color: theme.accent, width: 3.5 }
  });

  // Subtitle
  slide.addText("答辩问答预案 V1.0.0", {
    x: 0.5, y: 3.55, w: 9, h: 0.45,
    fontSize: 22, fontFace: "Microsoft YaHei",
    color: theme.bg, bold: false,
    align: "center", valign: "middle"
  });

  // Sub-subtitle
  slide.addText("含 10 大 Q&A + 通用速答模板", {
    x: 0.5, y: 4.0, w: 9, h: 0.4,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  // Page badge — keep visible on dark background
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("17", {
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
  pres.writeFile({ fileName: "slide-17-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
