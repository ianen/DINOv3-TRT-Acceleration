const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 12,
  title: '工程方法学创新（3 项）'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("工程方法学创新（3 项）", {
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
  slide.addText("DISCUSSION · Methodology", {
    x: 7.0, y: 0.32, w: 2.8, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("可复用的研究工程实践 — 超越 DINOv3 单一项目", {
    x: 0.4, y: 0.95, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // Card layout configuration
  const cardY = 1.5;
  const cardH = 3.4;
  const cardW = 3.0;
  const cardGap = 0.15;
  const cardStartX = 0.4;
  const cardBg = "eef2f7"; // slightly darker than bg
  const numBadgeH = 0.55;

  // Card data
  const cards = [
    {
      num: "01",
      header: "Pure-Python testing",
      bullets: [
        "layer_precision / onnx_qdq_stripper / strip_planner 全本地可测",
        "271 tests / 111 源文件",
        "GPU-free dev workflow"
      ]
    },
    {
      num: "02",
      header: "Bidirectional remote-sync",
      bullets: [
        "--pull-reports 文本产物反向回拉",
        "绕开 cpolar SSH scp 不稳定",
        "macOS ↔ Windows 双向闭环"
      ]
    },
    {
      num: "03",
      header: "Unified figure regen",
      bullets: [
        "build_all_figures.py 4 子系统统一入口",
        "figures_index.json 跨重生 diff 验证",
        "可复现产物 manifest"
      ]
    }
  ];

  cards.forEach((card, i) => {
    const cardX = cardStartX + i * (cardW + cardGap);

    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: cardY, w: cardW, h: cardH,
      fill: { color: cardBg },
      line: { color: theme.secondary, width: 1.25 }
    });

    // Number badge at top
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: cardY, w: cardW, h: numBadgeH,
      fill: { color: theme.primary },
      line: { type: "none" }
    });
    slide.addText(card.num, {
      x: cardX + 0.15, y: cardY, w: 0.6, h: numBadgeH,
      fontSize: 18, fontFace: "Arial",
      color: theme.accent, bold: true,
      align: "left", valign: "middle"
    });
    slide.addText(card.header, {
      x: cardX + 0.7, y: cardY, w: cardW - 0.85, h: numBadgeH,
      fontSize: 13, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "left", valign: "middle"
    });

    // Bullets in card body
    const bulletStartY = cardY + numBadgeH + 0.2;
    const bulletGap = 0.75;

    card.bullets.forEach((b, j) => {
      const by = bulletStartY + j * bulletGap;

      // Small accent dot
      slide.addShape(pres.shapes.OVAL, {
        x: cardX + 0.2, y: by + 0.1, w: 0.12, h: 0.12,
        fill: { color: theme.accent },
        line: { type: "none" }
      });

      // Bullet text
      slide.addText(b, {
        x: cardX + 0.4, y: by, w: cardW - 0.55, h: 0.65,
        fontSize: 11, fontFace: "Microsoft YaHei",
        color: theme.secondary, bold: false,
        align: "left", valign: "top"
      });
    });
  });

  // Footer label
  slide.addText("三项实践全部 open-source 在仓库 — 可复用作为研究型工程模板", {
    x: 0.4, y: 5.0, w: 8.8, h: 0.3,
    fontSize: 11, fontFace: "Microsoft YaHei",
    color: theme.light, italic: true,
    align: "center", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("12", {
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
  pres.writeFile({ fileName: "slide-12-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
