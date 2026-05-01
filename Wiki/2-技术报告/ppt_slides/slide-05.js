const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 5,
  title: '多分辨率 + Python/C++ 一致性'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("多分辨率 + Python/C++ 一致性", {
    x: 0.4, y: 0.25, w: 7.5, h: 0.55,
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
  slide.addText("Method · Multi-resolution + Parity", {
    x: 6.6, y: 0.32, w: 2.8, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Bullets list (left column)
  const bullets = [
    { tag: "01", text: "多分辨率：r224 / r336 / r518 各自独立 engine" },
    { tag: "02", text: "r518 batch 8 用 min=1, opt=4, max=8 profile，独立 timing cache" },
    { tag: "03", text: "C++ runtime：MSVC + CUDA + TRT 10.13.2.6 + RAII engine wrapper" },
    { tag: "04", text: "跨语言 parity：deterministic sine input → bit-identical (max_abs=0, cos=1.0)" }
  ];

  const startY = 1.1;
  const gap = 0.65;

  bullets.forEach((b, i) => {
    const y = startY + i * gap;

    // Tag pill
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.4, y: y + 0.05, w: 0.5, h: 0.4,
      fill: { color: theme.secondary },
      line: { type: "none" },
      rectRadius: 0.05
    });
    slide.addText(b.tag, {
      x: 0.4, y: y + 0.05, w: 0.5, h: 0.4,
      fontSize: 12, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });

    // Bullet text
    slide.addText(b.text, {
      x: 1.0, y: y, w: 4.6, h: 0.5,
      fontSize: 13.5, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Right side: 3 stacked horizontal bars showing resolution × token count
  const barLeft = 5.9;
  const barTopY = 1.15;
  const barRowH = 0.85;
  const labelW = 0.85;
  const barTrackX = barLeft + labelW;
  const barTrackW = 3.3;

  // Token counts and bar widths (proportional to token count, max 1025)
  const resData = [
    { label: "r224", tokens: 197,  shape: "224×224", barRatio: 197 / 1025 },
    { label: "r336", tokens: 442,  shape: "336×336", barRatio: 442 / 1025 },
    { label: "r518", tokens: 1025, shape: "518×518", barRatio: 1025 / 1025 }
  ];

  // Visual title
  slide.addText("Resolution × Token Count", {
    x: barLeft, y: 0.92, w: 4.0, h: 0.3,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, bold: true,
    align: "left", valign: "middle",
    charSpacing: 3
  });

  resData.forEach((r, i) => {
    const y = barTopY + i * barRowH;

    // Resolution label (left)
    slide.addText(r.label, {
      x: barLeft, y: y, w: labelW, h: 0.4,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true,
      align: "left", valign: "middle"
    });

    // Bar background (track)
    slide.addShape(pres.shapes.RECTANGLE, {
      x: barTrackX, y: y + 0.12, w: barTrackW, h: 0.18,
      fill: { color: "FFFFFF" },
      line: { color: theme.light, width: 0.4 }
    });

    // Bar fill (proportional)
    const fillW = barTrackW * r.barRatio;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: barTrackX, y: y + 0.12, w: fillW, h: 0.18,
      fill: { color: theme.accent },
      line: { type: "none" }
    });

    // Token count label (under bar)
    slide.addText(r.tokens + " tokens · " + r.shape, {
      x: barTrackX, y: y + 0.32, w: barTrackW, h: 0.32,
      fontSize: 11, fontFace: "Arial",
      color: theme.secondary, italic: false,
      align: "left", valign: "middle"
    });
  });

  // Bottom note for parity
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.3, w: 9.0, h: 0.55,
    fill: { color: theme.primary },
    line: { type: "none" }
  });
  slide.addText("Parity 验证已闭合：max_abs_error = 0,  cosine = 1.0  ·  超出 V1.0.1 G3 最严档", {
    x: 0.55, y: 4.3, w: 8.7, h: 0.55,
    fontSize: 13, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "left", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("05", {
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
  pres.writeFile({ fileName: "slide-05-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
