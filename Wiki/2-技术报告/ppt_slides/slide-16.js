const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 16,
  title: 'Reproducibility & License'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Title
  slide.addText("Reproducibility & License", {
    x: 0.4, y: 0.25, w: 7.5, h: 0.55,
    fontSize: 28, fontFace: "Arial",
    color: theme.primary, bold: true,
    align: "left", valign: "middle"
  });

  // Title underline
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 0.85, w: 1.6, h: 0,
    line: { color: theme.accent, width: 2.5 }
  });

  // Section label
  slide.addText("DELIVERABLE · 可复现 · 合规", {
    x: 7.0, y: 0.32, w: 2.6, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "right", valign: "middle"
  });

  // Subtitle
  slide.addText("两栏：复现工作流 + License 与进度记录", {
    x: 0.4, y: 0.95, w: 8, h: 0.35,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: theme.secondary, bold: false,
    align: "left", valign: "middle"
  });

  // Left column card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.4, w: 4.55, h: 3.4,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.5 }
  });

  // Left column header bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.4, w: 4.55, h: 0.5,
    fill: { color: theme.primary },
    line: { type: "none" }
  });
  slide.addText("可复现工作流", {
    x: 0.4, y: 1.4, w: 4.55, h: 0.5,
    fontSize: 15, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  const leftBullets = [
    "一键 PowerShell：scripts/run_formal_hf_pipeline_windows.ps1",
    "Figures 重生：scripts/build_all_figures.py --allow-missing",
    "Atomic SHA256 manifest 自动 exclude 自身 — 419+ 文件"
  ];

  leftBullets.forEach((text, i) => {
    const y = 2.05 + i * 0.85;

    // Bullet dot
    slide.addShape(pres.shapes.OVAL, {
      x: 0.6, y: y + 0.1, w: 0.16, h: 0.16,
      fill: { color: theme.accent },
      line: { type: "none" }
    });

    slide.addText(text, {
      x: 0.85, y: y, w: 4.0, h: 0.7,
      fontSize: 11.5, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Right column card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.05, y: 1.4, w: 4.55, h: 3.4,
    fill: { color: "FFFFFF" },
    line: { color: theme.light, width: 0.5 }
  });

  // Right column header bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.05, y: 1.4, w: 4.55, h: 0.5,
    fill: { color: theme.secondary },
    line: { type: "none" }
  });
  slide.addText("License & 进度记录", {
    x: 5.05, y: 1.4, w: 4.55, h: 0.5,
    fontSize: 15, fontFace: "Microsoft YaHei",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  const rightBullets = [
    "DINOv3 License 副本 + Built with DINOv3 标注全部就位",
    "33+ 篇心跳记录每一步可追溯",
    "271 pytest passing / 111 Python 源文件 ruff/mypy 全绿"
  ];

  rightBullets.forEach((text, i) => {
    const y = 2.05 + i * 0.85;

    // Bullet dot
    slide.addShape(pres.shapes.OVAL, {
      x: 5.25, y: y + 0.1, w: 0.16, h: 0.16,
      fill: { color: theme.accent },
      line: { type: "none" }
    });

    slide.addText(text, {
      x: 5.5, y: y, w: 4.0, h: 0.7,
      fontSize: 11.5, fontFace: "Microsoft YaHei",
      color: theme.primary, bold: false,
      align: "left", valign: "middle"
    });
  });

  // Bottom small text
  slide.addText("License 副本：LICENSES/DINOv3_LICENSE.md  |  进度记录：Wiki/0-项目计划/milestones/M1-progress.md", {
    x: 0.4, y: 4.95, w: 8.5, h: 0.32,
    fontSize: 10.5, fontFace: "Arial",
    color: theme.light, italic: true,
    align: "left", valign: "middle"
  });

  // Page badge
  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent },
    line: { type: "none" }
  });
  slide.addText("16", {
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
  pres.writeFile({ fileName: "slide-16-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
