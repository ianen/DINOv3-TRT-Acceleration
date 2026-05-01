const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 2,
  title: '1-slide TL;DR'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title bar
  slide.addText("1-slide TL;DR", {
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
  slide.addText("Executive Summary", {
    x: 7.0, y: 0.32, w: 2.3, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Bullets — left column
  const bullets = [
    { num: "01", text: "BF16 prefer 顶点 3.86× speedup（r518 b8 trtexec），三档 cos ≥ 0.998" },
    { num: "02", text: "INT8 全路径 sensitivity 已闭合 — Root cause：前段累积量化噪声" },
    { num: "03", text: "V1.3 方向：QAT 量化感知 fine-tuning（ADR-011 Proposed）" },
    { num: "04", text: "Python ↔ C++ 三档 batch=1 bit-identical" },
    { num: "05", text: "56 行 benchmark matrix + 8 张 SVG + 271 tests + 111 源文件" }
  ];

  const bulletStartY = 1.05;
  const bulletGap = 0.55;

  bullets.forEach((b, i) => {
    const y = bulletStartY + i * bulletGap;

    // Number badge (square with primary fill)
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: y, w: 0.45, h: 0.4,
      fill: { color: theme.secondary },
      line: { type: "none" }
    });
    slide.addText(b.num, {
      x: 0.4, y: y, w: 0.45, h: 0.4,
      fontSize: 12, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });

    // Bullet text
    slide.addText(b.text, {
      x: 0.95, y: y - 0.02, w: 4.55, h: 0.45,
      fontSize: 14, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Right side: thumbnail of cosine vs speedup tradeoff
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.5, y: 1.05, w: 4.2, h: 2.5,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.75 }
  });

  slide.addImage({
    path: "imgs/benchmark_bf16_vs_int8_tradeoff.svg",
    x: 5.55, y: 1.1, w: 4.1, h: 2.4
  });

  // Thumbnail caption
  slide.addText("BF16 vs INT8 · cosine 与 speedup 折中", {
    x: 5.5, y: 3.6, w: 4.2, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  // Footer reference text near thumbnail
  slide.addText("详见 Page 8", {
    x: 5.5, y: 3.95, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("02", {
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
  pres.writeFile({ fileName: "slide-02-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
