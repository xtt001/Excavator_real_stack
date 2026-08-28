#!/usr/bin/env python3
"""Build the conditioned-ACT weekly review as Markdown plus native SVG figures."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "group_meeting_conditioned_act_md_svg_20260828"
SVG_DIR = OUT_DIR / "svg"
MD_PATH = OUT_DIR / "index.md"

W = 1600

NAVY = "#102A43"
NAVY_2 = "#183B56"
BLUE = "#2F6FED"
TEAL = "#12A594"
ORANGE = "#F59E0B"
PURPLE = "#8356E8"
RED = "#D94848"
INK = "#172B4D"
MUTED = "#62748A"
PALE = "#F4F7FB"
PALE_BLUE = "#EAF1FF"
PALE_TEAL = "#E8F8F5"
PALE_ORANGE = "#FFF5DF"
PALE_PURPLE = "#F1EBFF"
WHITE = "#FFFFFF"
LINE = "#D5E0EA"


def t(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 28,
    fill: str = INK,
    weight: int = 400,
    anchor: str = "start",
    opacity: float | None = None,
    css_class: str | None = None,
) -> str:
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
        f'font-weight="{weight}"',
        f'text-anchor="{anchor}"',
    ]
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    if css_class:
        attrs.append(f'class="{escape(css_class)}"')
    return f"<text {' '.join(attrs)}>{escape(value)}</text>"


def lines(
    x: float,
    y: float,
    values: list[str] | tuple[str, ...],
    *,
    size: int = 28,
    line_height: int = 42,
    fill: str = INK,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return "\n".join(
        t(
            x,
            y + index * line_height,
            value,
            size=size,
            fill=fill,
            weight=weight,
            anchor=anchor,
        )
        for index, value in enumerate(values)
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = WHITE,
    stroke: str = LINE,
    stroke_width: int = 2,
    radius: int = 24,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def circle(x: float, y: float, radius: float, *, fill: str) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"/>'


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = LINE,
    width: int = 3,
    arrow: bool = False,
    dash: str | None = None,
) -> str:
    attrs = [
        f'x1="{x1}"',
        f'y1="{y1}"',
        f'x2="{x2}"',
        f'y2="{y2}"',
        f'stroke="{stroke}"',
        f'stroke-width="{width}"',
        'stroke-linecap="round"',
    ]
    if arrow:
        attrs.append('marker-end="url(#arrowhead)"')
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return f"<line {' '.join(attrs)}/>"


def pill(
    x: float,
    y: float,
    width: float,
    label: str,
    *,
    fill: str,
    text_fill: str = WHITE,
    height: int = 52,
    size: int = 23,
) -> str:
    return "\n".join(
        [
            rect(x, y, width, height, fill=fill, stroke=fill, radius=height // 2),
            t(
                x + width / 2,
                y + height / 2 + size * 0.36,
                label,
                size=size,
                fill=text_fill,
                weight=700,
                anchor="middle",
            ),
        ]
    )


def bullet(
    x: float,
    y: float,
    value: str,
    *,
    colour: str,
    size: int = 25,
    weight: int = 400,
) -> str:
    return "\n".join(
        [
            circle(x, y - 8, 7, fill=colour),
            t(x + 25, y, value, size=size, fill=INK, weight=weight),
        ]
    )


def document(
    *,
    title: str,
    description: str,
    height: int,
    body: list[str],
    dark: bool = False,
) -> str:
    background = NAVY if dark else PALE
    foreground = WHITE if dark else NAVY
    content = "\n".join(body)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" role="img" aria-labelledby="svg-title svg-desc">
  <title id="svg-title">{escape(title)}</title>
  <desc id="svg-desc">{escape(description)}</desc>
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#9FB1C2"/>
    </marker>
    <style>
      text {{ font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", Arial, sans-serif; }}
    </style>
  </defs>
  <rect width="{W}" height="{height}" fill="{background}"/>
  <rect width="{W}" height="12" fill="{BLUE}"/>
  <text x="78" y="68" font-family="Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, Arial, sans-serif" font-size="19" font-weight="700" fill="{BLUE if not dark else '#A9C1D6'}">本周组会 · 2026.08.28</text>
  <text x="78" y="128" font-family="Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, Arial, sans-serif" font-size="45" font-weight="700" fill="{foreground}">{escape(title)}</text>
  <line x1="78" y1="154" x2="1522" y2="154" stroke="{'#35556E' if dark else LINE}" stroke-width="2"/>
{content}
</svg>
'''


def save_svg(name: str, content: str) -> None:
    path = SVG_DIR / name
    path.write_text(content, encoding="utf-8")


def build_work_map() -> None:
    body: list[str] = []
    body.append(t(78, 195, "左右目标响应与原有一铲能力共同作为上机验收条件。", size=27, fill=MUTED))
    tasks = [
        (70, TEAL, "新增左右目标响应", ("左区继续有效回转", "右区回零停止")),
        (570, ORANGE, "保留原一铲能力", ("保持普通挖掘与回程", "机器不动时仍能跨过死区")),
        (1070, BLUE, "共同验收", ("开发验证集和独立留出集", "两项任务的标准必须一起通过")),
    ]
    for x, colour, title, detail in tasks:
        body.append(rect(x, 230, 460, 156, fill=WHITE))
        body.append(f'<rect x="{x}" y="230" width="9" height="156" fill="{colour}"/>')
        body.append(t(x + 32, 276, title, size=27, fill=colour, weight=700))
        body.append(lines(x + 32, 324, detail, size=22, line_height=36, fill=INK, weight=500))
    body.append(t(78, 450, "六个阶段", size=27, fill=NAVY, weight=700))
    stages = [
        (70, BLUE, "1", "确认数据事实", ("目标差异只在", "释放阶段")),
        (310, RED, "2", "验证三条路线", ("比较三条路线", "的动作结果")),
        (550, TEAL, "3", "冻结最终损失", ("确定阶段、动作", "余量和权重")),
        (790, PURPLE, "4", "训练并选择候选", ("按动作标准选择", "不按最低损失")),
        (1030, ORANGE, "5", "运行离线测试", ("完整序列、左右目标", "和机器不动")),
        (1270, NAVY_2, "6", "分别回答两项任务", ("给出结论和", "上机证据边界")),
    ]
    for x, colour, number, title, detail in stages:
        body.append(rect(x, 490, 210, 158, fill=WHITE))
        body.append(circle(x + 32, 526, 21, fill=colour))
        body.append(t(x + 32, 534, number, size=22, fill=WHITE, weight=700, anchor="middle"))
        body.append(t(x + 62, 535, title, size=21, fill=NAVY, weight=700))
        body.append(lines(x + 20, 585, detail, size=17, line_height=28, fill=MUTED, weight=500))
    for x1, x2 in ((280, 310), (520, 550), (760, 790), (1000, 1030), (1240, 1270)):
        body.append(line(x1 + 2, 569, x2 - 5, 569, stroke="#9FB1C2", width=4, arrow=True))
    body.append(rect(210, 690, 1180, 64, fill=NAVY, stroke=NAVY, radius=18))
    body.append(t(800, 732, "两项任务贯穿数据分析、训练和离线验收。", size=24, fill=WHITE, weight=700, anchor="middle"))
    save_svg(
        "00-work-map.svg",
        document(
            title="本周任务与时间线",
            description="任务线要求新增左右目标响应并保留原一铲能力，时间线依次经过数据确认、路线验证、最终设计、训练选择、离线测试和结论。",
            height=790,
            body=body,
        ),
    )


def build_conclusion() -> None:
    body: list[str] = []
    body.append(t(78, 198, "新的回转释放损失同时建立了两类能力。", size=27, fill=MUTED))
    cards = [
        (78, TEAL, "99.0%", "开发验证集", "左右目标成对命中"),
        (579, BLUE, "92.7%", "独立留出集", "左右目标成对命中"),
        (1080, ORANGE, "不低于原模型", "两套数据", "冻结状态下跨越死区"),
    ]
    for x, colour, value, label, detail in cards:
        body.append(rect(x, 235, 442, 222, fill=WHITE))
        body.append(f'<rect x="{x}" y="235" width="10" height="222" fill="{colour}"/>')
        body.append(t(x + 221, 315, value, size=50 if x < 1000 else 41, fill=colour, weight=700, anchor="middle"))
        body.append(t(x + 221, 370, label, size=24, fill=MUTED, weight=700, anchor="middle"))
        body.append(t(x + 221, 412, detail, size=25, fill=INK, anchor="middle"))
    body.append(rect(78, 500, 1444, 225, fill=WHITE))
    body.append(t(115, 548, "明确的动作要求", size=29, fill=NAVY, weight=700))
    rows = [
        (595, TEAL, "左区目标", "继续负向回转，动作达到机械有效幅值"),
        (652, BLUE, "右区目标", "在右区回零停止，不再继续回转"),
        (709, ORANGE, "死区能力", "冻结观测后仍能自行再次输出越过死区的动作"),
    ]
    for y, colour, label, detail in rows:
        body.append(pill(115, y - 35, 154, label, fill=colour, height=46, size=21))
        body.append(t(300, y, detail, size=26, fill=INK, weight=600))
    save_svg(
        "01-conclusion.svg",
        document(
            title="左右目标响应与死区能力",
            description="新损失在两套离线数据上建立左右目标响应，同时保留原模型跨越机械死区的能力。",
            height=780,
            body=body,
        ),
    )


def build_data_distribution() -> None:
    body: list[str] = []
    body.append(t(78, 198, "完整周期中，左右目标共享绝大多数动作，真正分叉只发生在回转释放阶段。", size=27, fill=MUTED))
    body.append(t(95, 255, "完整动作链", size=27, fill=NAVY, weight=700))
    segments = [
        (95, 180, "挖掘", "#B9C7D5"),
        (285, 240, "抬升与装载", "#9EB1C4"),
        (535, 300, "正向回转卸料", "#7894AD"),
        (845, 350, "共同负向回转", "#4D789D"),
        (1205, 300, "目标释放", ORANGE),
    ]
    for x, width, label, colour in segments:
        body.append(rect(x, 290, width, 82, fill=colour, stroke=colour, radius=16))
        body.append(t(x + width / 2, 342, label, size=22, fill=WHITE, weight=700, anchor="middle"))
    body.append(t(95, 438, "训练时间步分布", size=27, fill=NAVY, weight=700))
    bar_x, bar_y, bar_width = 95, 470, 1410
    common_width = int(bar_width * 0.9415)
    body.append(rect(bar_x, bar_y, bar_width, 78, fill="#DCE5ED", stroke="#DCE5ED", radius=18))
    body.append(rect(bar_x + common_width, bar_y, bar_width - common_width, 78, fill=ORANGE, stroke=ORANGE, radius=18))
    body.append(t(bar_x + common_width / 2, bar_y + 50, "共同动作阶段 94.15%", size=27, fill=NAVY, weight=700, anchor="middle"))
    body.append(line(bar_x + common_width, bar_y - 10, bar_x + common_width, bar_y + 90, stroke=WHITE, width=4))
    body.append(t(1505, 590, "目标相关区间 5.85%", size=25, fill=ORANGE, weight=700, anchor="end"))
    body.append(line(1432, 554, 1464, 574, stroke=ORANGE, width=3))
    body.append(rect(95, 618, 1410, 92, fill=WHITE))
    body.append(t(130, 674, "严格可监督起点仅约 1.86%", size=31, fill=RED, weight=700))
    body.append(t(700, 674, "全周期平均训练会让共同动作淹没左右目标产生的训练信号。", size=28, fill=INK, weight=600))
    save_svg(
        "02-data-distribution.svg",
        document(
            title="目标信息集中在回转释放阶段",
            description="左右目标共享百分之九十四以上的周期动作，目标差异集中在最后的回转释放阶段。",
            height=760,
            body=body,
        ),
    )


def build_failed_designs() -> None:
    body: list[str] = []
    body.append(t(78, 198, "三条路线分别暴露出动作差不足、假设与数据不符和死区余量不足。", size=27, fill=MUTED))
    columns = [
        (150, "方案是什么"),
        (575, "原本想解决什么"),
        (850, "实际结果"),
        (1320, "结论"),
    ]
    body.append(rect(70, 225, 1460, 58, fill=NAVY, stroke=NAVY, radius=14))
    for x, label in columns:
        body.append(t(x, 263, label, size=22, fill=WHITE, weight=700))
    rows = [
        (
            298,
            RED,
            "1",
            ("给动作预测加左右分类", "测试权重 0.5、1、2、5"),
            ("让模型注意到", "当前左右目标"),
            ("权重 1、2、5 分类达到 100%", "回转动作差最大仅 0.014–0.016"),
            ("差值远小于", "负向死区 0.721"),
        ),
        (
            460,
            ORANGE,
            "2",
            ("规定左区向负方向", "右区向正方向"),
            ("用相反方向", "强行拉开两类动作"),
            ("数据表明两类回程都主要向负方向", "区别是何时停止，不是方向相反"),
            ("基本假设与", "真实动作不符"),
        ),
        (
            622,
            PURPLE,
            "3",
            ("左区动作 ≤ −0.721", "右区动作 = 0"),
            ("直接把左右目标", "写进动作要求"),
            ("零损失允许动作刚好压在死区边缘", "短训成对命中约 67%–74%"),
            ("没有余量，也未达到", "预定的 80% 标准"),
        ),
    ]
    for y, colour, number, design, purpose, result, decision in rows:
        body.append(rect(70, y, 1460, 145, fill=WHITE))
        body.append(rect(70, y, 62, 145, fill=colour, stroke=colour, radius=18))
        body.append(t(101, y + 83, number, size=31, fill=WHITE, weight=700, anchor="middle"))
        body.append(lines(150, y + 49, design, size=21, line_height=34, fill=INK, weight=700))
        body.append(lines(575, y + 49, purpose, size=20, line_height=34, fill=MUTED, weight=500))
        body.append(lines(850, y + 45, result, size=20, line_height=35, fill=INK, weight=500))
        body.append(lines(1320, y + 49, decision, size=19, line_height=33, fill=colour, weight=700))
    body.append(rect(70, 800, 1460, 76, fill=PALE_TEAL, stroke=PALE_TEAL, radius=18))
    body.append(pill(95, 815, 150, "最终修改", fill=TEAL, height=46, size=20))
    body.append(t(275, 847, "只在释放阶段约束动作：左区 ≤ −0.799，右区 = 0，并要求两者相差至少 0.799。", size=24, fill=INK, weight=700))
    save_svg(
        "03-failed-designs.svg",
        document(
            title="三条训练路线",
            description="第一条只做目标分类，第二条强制左右方向相反，第三条把动作压在死区边缘；它们分别因动作差不足、数据假设错误和缺少动作余量而未采用。",
            height=920,
            body=body,
        ),
    )


def build_method() -> None:
    body: list[str] = []
    nodes = [
        (85, 220, 290, "图像 + 关节状态", PALE_BLUE, BLUE),
        (445, 220, 270, "只选择释放阶段 M(t)", PALE_ORANGE, ORANGE),
        (785, 220, 280, "左区 / 右区目标", PALE_TEAL, TEAL),
        (1135, 220, 380, "模型直接输出的动作序列", PALE_PURPLE, PURPLE),
    ]
    for x, y, width, label, background, colour in nodes:
        body.append(rect(x, y, width, 98, fill=background, stroke=background))
        body.append(t(x + width / 2, y + 61, label, size=25, fill=colour, weight=700, anchor="middle"))
    for x1, x2 in ((375, 445), (715, 785), (1065, 1135)):
        body.append(line(x1 + 10, 269, x2 - 12, 269, stroke="#A9B9C8", width=5, arrow=True))
    body.append(t(85, 382, "回转释放损失", size=29, fill=NAVY, weight=700))
    body.append(rect(85, 410, 1430, 108, fill=NAVY, stroke=NAVY, radius=22))
    formula = "L_release = max(0, m + a_left) + |a_right| + max(0, m − (a_right − a_left))"
    body.append(t(800, 478, formula, size=31, fill=WHITE, weight=700, anchor="middle"))
    implications = [
        (85, TEAL, "左区目标", "a_left ≤ −m", "继续回转并越过死区"),
        (565, BLUE, "右区目标", "a_right = 0", "在右区回零停止"),
        (1045, ORANGE, "动作间隔", "a_right − a_left ≥ m", "为泛化误差预留余量"),
    ]
    for x, colour, label, value, detail in implications:
        body.append(rect(x, 560, 430, 132, fill=WHITE))
        body.append(pill(x + 28, 584, 138, label, fill=colour, height=44, size=20))
        body.append(t(x + 195, 616, value, size=25, fill=colour, weight=700))
        body.append(t(x + 28, 665, detail, size=23, fill=INK, weight=600))
    parameters = [
        ("决策区", "0.111–0.393 弧度"),
        ("负向死区", "0.721"),
        ("动作余量 m", "0.799"),
        ("持续窗口", "5 帧 / 0.25 秒"),
        ("损失权重", "0.503"),
    ]
    card_width = 274
    for index, (label, value) in enumerate(parameters):
        x = 85 + index * 286
        body.append(rect(x, 728, card_width, 96, fill=WHITE))
        body.append(t(x + 20, 764, label, size=20, fill=MUTED, weight=700))
        body.append(t(x + 20, 805, value, size=23, fill=BLUE, weight=700))
    body.append(t(800, 870, "共同阶段仍由原有模仿训练学习；新增要求只作用于目标真正改变动作的释放区间。", size=25, fill=RED, weight=700, anchor="middle"))
    save_svg(
        "04-method.svg",
        document(
            title="最终训练方案",
            description="训练只选择目标释放区间，损失直接要求左区继续回转、右区停止，并保留动作间隔。",
            height=910,
            body=body,
        ),
    )


def build_theory() -> None:
    body: list[str] = []
    body.append(rect(70, 200, 700, 410, fill=WHITE))
    body.append(t(110, 255, "Keyframe-Focused Visual Imitation Learning", size=31, fill=NAVY, weight=700))
    body.append(t(110, 297, "Wen et al. · ICML 2021", size=22, fill=BLUE, weight=700))
    body.append(line(110, 325, 730, 325, stroke=LINE, width=2))
    findings = [
        "长序列中真正改变行为的动作变化点很少",
        "均匀采样容易复制上一动作或依赖惯性",
        "提高关键帧权重能改善视觉模仿控制",
    ]
    for index, value in enumerate(findings):
        body.append(bullet(115, 382 + index * 72, value, colour=BLUE, size=24))
    body.append(rect(830, 200, 700, 410, fill=WHITE))
    body.append(t(870, 255, "与本项目的对应", size=31, fill=NAVY, weight=700))
    mappings = [
        ("稀少动作变化点", "左右目标的回转释放决策"),
        ("重复动作占多数", "共同挖掘、装载与卸料阶段"),
        ("关键帧重加权", "目标阶段附加采样与掩码"),
        ("复制与惯性", "忽略目标并回到平均动作"),
    ]
    for index, (left, right) in enumerate(mappings):
        y = 335 + index * 64
        body.append(t(875, y, left, size=23, fill=MUTED))
        body.append(t(1135, y, "→", size=25, fill=ORANGE, weight=700, anchor="middle"))
        body.append(t(1180, y, right, size=23, fill=INK, weight=700))
    body.append(t(80, 665, "其他理论支撑", size=27, fill=NAVY, weight=700))
    references = [
        (80, BLUE, "任务分段", "TACO · CompILE", "长周期按责任边界对齐"),
        (460, TEAL, "目标回溯", "HER · GCSL", "用实际结果构造目标监督"),
        (840, ORANGE, "条件模仿", "Conditional Imitation", "高层目标控制同一低层策略"),
        (1220, PURPLE, "机械合同", "本项目数据", "把理论原则落到执行器动作"),
    ]
    for x, colour, title, ref, detail in references:
        body.append(rect(x, 700, 300, 112, fill=WHITE))
        body.append(f'<rect x="{x}" y="700" width="8" height="112" fill="{colour}"/>')
        body.append(t(x + 28, 738, title, size=23, fill=colour, weight=700))
        body.append(t(x + 28, 770, ref, size=19, fill=MUTED, weight=700))
        body.append(t(x + 28, 798, detail, size=18, fill=INK))
    body.append(t(800, 866, "关键帧研究解释了释放阶段为何需要更高训练密度；真机数据提供动作幅值与死区参数。", size=24, fill=RED, weight=700, anchor="middle"))
    save_svg(
        "05-theory.svg",
        document(
            title="关键动作变化与训练重点",
            description="关键帧模仿学习支持提高稀疏动作变化点的训练权重，任务分段、目标回溯和条件模仿提供补充支持。",
            height=900,
            body=body,
        ),
    )


def build_offline_test() -> None:
    body: list[str] = []
    body.append(rect(78, 182, 1444, 74, fill=PALE_BLUE, stroke=PALE_BLUE, radius=18))
    body.append(pill(105, 195, 140, "测试单位", fill=BLUE, height=46, size=20))
    body.append(t(270, 228, "一条包含 3、4 或 5 铲的完整实机记录；从第一个目标开始，一直运行到目标序列结束。", size=24, fill=INK, weight=600))
    body.append(t(78, 304, "一次完整回放的基本流程", size=27, fill=NAVY, weight=700))
    nodes = [
        (70, 190, "读取完整序列", "按实机记录顺序逐铲读取"),
        (280, 190, "提交当前目标", "每铲开始只提交这一个目标"),
        (490, 210, "清除上一铲记录", "旧动作预测和内部历史归零"),
        (720, 210, "逐帧读取观测", "四路图像、位置、速度"),
        (950, 170, "模型给动作", "保存模型原始输出"),
        (1140, 170, "经过动作保护", "保存保护后的动作"),
        (1330, 200, "按阶段记分", "挖掘→回程→释放→稳定"),
    ]
    for x, width, label, detail in nodes:
        body.append(rect(x, 340, width, 102, fill=WHITE))
        body.append(t(x + width / 2, 382, label, size=22, fill=NAVY, weight=700, anchor="middle"))
        body.append(t(x + width / 2, 418, detail, size=17, fill=MUTED, anchor="middle"))
    for x1, x2 in ((260, 280), (470, 490), (700, 720), (930, 950), (1120, 1140), (1310, 1330)):
        body.append(line(x1 + 3, 391, x2 - 6, 391, stroke="#9FB1C2", width=4, arrow=True))
    branches = [
        (
            70,
            BLUE,
            "① 每一帧都做的基本检查",
            [
                "并排保存模型原始动作、保护后动作和专家动作",
                "根据真实最高点、目标区入口和稳定窗划分阶段",
                "分别计算误差、方向、挖掘、回程和停止表现",
            ],
        ),
        (
            565,
            TEAL,
            "② 到目标决策区时做的对照",
            [
                "仅取最高点之后、训练合同支持的决策区观测",
                "同一图像和状态分别输入左区、右区目标",
                "另开不做历史动作平均的模型，只看当前一步",
            ],
        ),
        (
            1060,
            ORANGE,
            "③ 到动作刚越过死区时做的对照",
            [
                "在专家动作由死区内转为有效动作处选测试点",
                "每次从相同内部状态开始，连续判断二十次",
                "检查同轴同向再启动以及错误有效动作",
            ],
        ),
    ]
    for x, colour, title, items in branches:
        body.append(rect(x, 500, 470, 248, fill=WHITE))
        body.append(rect(x, 500, 470, 62, fill=colour, stroke=colour, radius=20))
        body.append(t(x + 235, 541, title, size=25, fill=WHITE, weight=700, anchor="middle"))
        for index, item in enumerate(items):
            body.append(bullet(x + 35, 607 + index * 57, item, colour=colour, size=20))
    body.append(rect(78, 790, 1444, 92, fill=NAVY, stroke=NAVY, radius=20))
    body.append(t(800, 826, "记录数据确认稳定回位后，结束当前一铲、提交下一个目标；重复直到整条目标序列结束。", size=24, fill=WHITE, weight=700, anchor="middle"))
    body.append(t(800, 859, "最后分别汇总开发验证集和独立留出集；完整动作、左右目标和机器不动三项标准必须一起通过。", size=21, fill="#D5E4EF", anchor="middle"))
    save_svg(
        "06-offline-test.svg",
        document(
            title="完整离线回放流程",
            description="测试以一条三到五铲的完整实机记录为单位，逐铲提交目标、逐帧读取真实观测、记录模型和保护后动作，并在稳定回位后进入下一铲。",
            height=930,
            body=body,
        ),
    )


def build_state_hold_test() -> None:
    body: list[str] = []
    body.append(t(78, 198, "这个测试专门复现一种闭环风险：动作没有越过死区，机器不动，模型下一次仍看到原状态。", size=26, fill=MUTED))
    steps = [
        (
            70,
            BLUE,
            "1. 选测试点",
            ("专家动作从死区内转为有效", "按轴和方向分别标记；排除无效样本"),
        ),
        (
            455,
            PURPLE,
            "2. 还原当时历史",
            ("逐帧读取测试点之前的真实观测", "还原模型对过去几步的内部记录"),
        ),
        (
            840,
            ORANGE,
            "3. 每次从同样状态开始",
            ("保存并恢复同一份模型内部状态", "固定图像、位置、目标，速度置零"),
        ),
        (
            1225,
            TEAL,
            "4. 让模型连续判断",
            ("同一观测调用策略二十次", "动作不推进状态，不借用专家下一帧"),
        ),
    ]
    for x, colour, title, detail in steps:
        body.append(rect(x, 235, 315, 150, fill=WHITE))
        body.append(rect(x, 235, 315, 54, fill=colour, stroke=colour, radius=20))
        body.append(t(x + 157.5, 272, title, size=24, fill=WHITE, weight=700, anchor="middle"))
        body.append(lines(x + 24, 327, detail, size=19, line_height=32, fill=INK, weight=500))
    for x1, x2 in ((385, 455), (770, 840), (1155, 1225)):
        body.append(line(x1 + 10, 310, x2 - 12, 310, stroke="#9FB1C2", width=4, arrow=True))
    body.append(t(80, 446, "对照方式", size=27, fill=NAVY, weight=700))
    body.append(rect(80, 474, 1440, 118, fill=WHITE))
    body.append(t(112, 520, "此前真实记录", size=21, fill=BLUE, weight=700))
    for index in range(6):
        body.append(circle(280 + index * 58, 513, 12, fill=BLUE if index < 5 else ORANGE))
        if index < 5:
            body.append(line(292 + index * 58, 513, 326 + index * 58, 513, stroke="#A9B9C8", width=3))
    body.append(t(580, 520, "测试点的模型状态", size=21, fill=ORANGE, weight=700))
    body.append(line(700, 513, 770, 513, stroke="#9FB1C2", width=4, arrow=True))
    body.append(t(800, 500, "对照 1：按一个测试轴和方向判断", size=20, fill=TEAL, weight=700))
    body.append(t(800, 536, "对照 2：同一帧有多个测试点时逐一重跑", size=20, fill=PURPLE, weight=700))
    body.append(line(1100, 513, 1170, 513, stroke="#9FB1C2", width=4, arrow=True))
    body.append(t(1200, 500, "每次测试前恢复相同状态", size=20, fill=NAVY, weight=700))
    body.append(t(1200, 536, "测试结束后再恢复并继续真实记录", size=19, fill=MUTED))
    body.append(t(80, 650, "评价指标", size=27, fill=NAVY, weight=700))
    criteria = [
        (80, BLUE, "5 次内同向有效", "只作短窗诊断，不作为最终硬门槛"),
        (450, TEAL, "20 次内同向有效", "机器没动时，能否自己重新发出有效动作"),
        (820, RED, "首步错误有效动作", "目标轴反向，或出现未允许的其他轴动作"),
        (1190, ORANGE, "启动 / 周期中分组", "第一个测试点单独报告，避免总体均值掩盖启动"),
    ]
    for x, colour, title, detail in criteria:
        body.append(rect(x, 680, 330, 102, fill=WHITE))
        body.append(f'<rect x="{x}" y="680" width="7" height="102" fill="{colour}"/>')
        body.append(t(x + 24, 720, title, size=21, fill=colour, weight=700))
        body.append(t(x + 24, 755, detail, size=17, fill=INK))
    body.append(rect(80, 820, 1440, 76, fill=NAVY, stroke=NAVY, radius=18))
    body.append(t(120, 851, "同一批数据、同一批测试点、同一张死区表，直接对比原一铲模型：", size=22, fill=WHITE, weight=700))
    body.append(t(900, 851, "开发 67.5% ≥ 55.4%，错误 26 ≤ 38", size=21, fill="#76E0D1", weight=700))
    body.append(t(900, 880, "留出 59.7% ≥ 58.4%，错误 29 ≤ 53", size=21, fill="#76E0D1", weight=700))
    save_svg(
        "06-state-hold.svg",
        document(
            title="机器不动测试",
            description="在专家动作刚从死区内变为有效动作的测试点，还原模型当时的历史，保持画面和姿态不变连续判断二十次，并与同一数据上的原模型比较。",
            height=940,
            body=body,
        ),
    )


def build_results() -> None:
    body: list[str] = []
    metrics = [
        (70, 350, TEAL, "99.0%", "开发验证集", "左右目标成对命中"),
        (450, 350, BLUE, "92.7%", "独立留出集", "左右目标成对命中"),
        (830, 350, ORANGE, "100%", "两套数据", "右区目标回零"),
        (1210, 320, NAVY_2, "0", "两套数据", "策略错误 / 保护触发"),
    ]
    for x, width, colour, value, label, detail in metrics:
        body.append(rect(x, 195, width, 190, fill=WHITE))
        body.append(f'<rect x="{x}" y="195" width="9" height="190" fill="{colour}"/>')
        body.append(t(x + width / 2, 270, value, size=51, fill=colour, weight=700, anchor="middle"))
        body.append(t(x + width / 2, 322, label, size=22, fill=MUTED, weight=700, anchor="middle"))
        body.append(t(x + width / 2, 356, detail, size=22, fill=INK, anchor="middle"))
    body.append(rect(70, 425, 720, 405, fill=WHITE))
    body.append(t(105, 474, "完整序列动作保持", size=29, fill=NAVY, weight=700))
    headers = [(120, "指标"), (430, "开发验证集"), (635, "独立留出集")]
    body.append(f'<rect x="105" y="505" width="650" height="56" fill="{PALE_BLUE}"/>')
    for x, value in headers:
        body.append(t(x, 542, value, size=21, fill=NAVY, weight=700))
    table_rows = [
        ("动作方向一致率", "92.9%", "91.6%"),
        ("挖掘阶段有效率", "23.4%", "24.7%"),
        ("回转阶段有效率", "66.8%", "60.9%"),
        ("平均绝对动作误差", "0.107", "0.114"),
    ]
    for index, row in enumerate(table_rows):
        y = 562 + index * 62
        if index % 2:
            body.append(f'<rect x="105" y="{y}" width="650" height="58" fill="#F5F8FB"/>')
        body.append(t(120, y + 38, row[0], size=22, fill=INK))
        body.append(t(430, y + 38, row[1], size=22, fill=INK))
        body.append(t(635, y + 38, row[2], size=22, fill=INK))
    body.append(rect(830, 425, 700, 405, fill=WHITE))
    body.append(t(865, 474, "机器没动时，20 次内重新给出同向有效动作", size=25, fill=NAVY, weight=700))
    groups = [
        ("开发验证集", 55.4, 67.5, 38, 26, 550),
        ("独立留出集", 58.4, 59.7, 53, 29, 690),
    ]
    for label, baseline, candidate, old_wrong, new_wrong, y in groups:
        body.append(t(865, y - 34, label, size=22, fill=MUTED, weight=700))
        body.append(t(1120, y - 34, f"第一次就给错方向或多余动作：{old_wrong} → {new_wrong}", size=17, fill=RED, weight=700))
        scale = 560 / 80.0
        body.append(rect(865, y, baseline * scale, 34, fill="#B6C7D8", stroke="#B6C7D8", radius=17))
        body.append(t(875 + baseline * scale, y + 25, f"原模型 {baseline:.1f}%", size=19, fill=NAVY, weight=700))
        body.append(rect(865, y + 48, candidate * scale, 34, fill=TEAL, stroke=TEAL, radius=17))
        body.append(t(875 + candidate * scale, y + 73, f"新模型 {candidate:.1f}%", size=19, fill=TEAL, weight=700))
    body.append(t(800, 875, "两项测试共同说明：模型开始听从左右目标，原有死区能力没有退化。", size=27, fill=TEAL, weight=700, anchor="middle"))
    save_svg(
        "07-results.svg",
        document(
            title="离线结果",
            description="左右目标成对命中率达到百分之九十九和百分之九十二点七，冻结状态下跨越死区的动作复现不低于原模型。",
            height=920,
            body=body,
        ),
    )


MARKDOWN = """# 从条件输入到可执行动作约束

**本周组会 · 2026-08-28**

> 新的回转释放损失让模型按照左右分区目标改变动作，同时保留原一铲模型的普通动作和死区能力。

![本周任务与时间线](svg/00-work-map.svg)

## 本周任务

第一项任务是增加左右目标响应。回程进入右区时，左区目标要求模型继续负向回转，右区目标要求模型回零停止。

第二项任务是保留原一铲模型的能力，包括普通挖掘、卸料、回程，以及机器不动时重新给出越过死区的动作。

两项任务共同构成模型的离线验收标准。

## 1. 确认目标影响动作的时刻

![目标信息集中在回转释放阶段](svg/02-data-distribution.svg)

训练集共有 40,595 个时间步和 60 个完整周期。左右分区目标共享约 94.15% 的动作，目标相关区间约占 5.85%，可用于左右对照的起点约占 1.86%。

数据表明，两类目标在挖掘、装载、卸料和大部分回程阶段都应该执行相同动作。差别只出现在回程经过右区、需要决定继续向左回转还是在右区停止的短暂阶段。

据此确定两个评估维度：

1. 同一个画面和姿态下，左右目标是否产生不同的回转命令。
2. 新增目标训练后，普通动作和死区能力是否保持。

## 2. 验证三条训练路线

![三条训练路线](svg/03-failed-designs.svg)

### 路线一　增加左右分类

具体做法是在模型的动作预测上增加左右分类，并分别测试 0.5、1、2、5 的损失权重。原本希望分类压力能迫使模型注意当前目标。

权重为 1、2、5 时，左右分类准确率达到 100%。同一个状态只换目标后，回转动作的最大差值约为 0.014–0.016，远低于负向机械死区 0.721。分类结果没有转化为机械有效动作。

### 路线二　规定相反方向

这条路线打算用方向直接拉开两类动作：左区使用负向回转，右区使用正向回转。

真实数据中，两类目标在回程阶段都主要使用负向动作。区别在于负向回转何时释放，而非动作方向相反。

### 路线三　把动作压到死区边缘

第三条路线直接要求左区动作不高于 −0.721、右区动作回到 0，并要求两者至少相差 0.721。

损失降到零时，左区动作仍可停在死区边缘。三组 80 轮短训的左右成对命中率为 67.0%、67.8% 和 74.3%，低于 80% 的验收标准。

### 最终调整

左区目标幅值改为训练数据中有效动作的中位数 0.799，右区保持回零，两者至少相差 0.799。该幅值为死区外的动作留出真实数据支持的余量。

## 3. 确定训练方案

![最终训练方案](svg/04-method.svg)

### 释放阶段取样

训练只选择回转最高点之后、位于右区决策范围内、且专家动作明确支持继续或停止的时刻。每个完整周期额外取一个这样的样本，避免候选点多的周期在训练中占据过多权重。

### 回转释放损失

同一个画面和姿态分别输入左区、右区目标。左区要求前五步继续输出不高于 −0.799 的回转动作；右区要求前五步回到 0；同时要求两组动作至少相差 0.799。

### 参数来源

右区决策范围、机械死区、0.799 的动作目标和五帧持续窗口来自训练数据。原训练损失与新增损失的初始量级给出权重 0.503。

### 保留原有一铲训练

原有模仿训练、死区损失和“机器不动时重新给动作”的训练采样仍然保留。新增损失只负责回转释放阶段，不重新规定整个周期的动作。

### 论文依据

![关键动作变化与训练重点](svg/05-theory.svg)

[Keyframe-Focused Visual Imitation Learning](https://proceedings.mlr.press/v139/wen21d.html) 指出，长序列中真正改变行为的动作变化点很少，均匀训练容易让大量重复动作淹没这些关键时刻。这与本项目把额外训练集中到回转释放阶段的做法一致。

[TACO](https://proceedings.mlr.press/v80/shiarlis18a.html) 和 [CompILE](https://proceedings.mlr.press/v97/kipf19a.html) 支持把长任务按动作阶段拆开学习。具体动作要求、死区阈值和余量由真机数据与执行器特性确定。

## 4. 训练与模型选择

训练运行 300 轮，最低验证损失出现在第 140 轮。多个保留版本继续接受动作验收。

第 199 轮模型同时通过左右目标、普通动作和机器不动三类标准，成为最终候选。

## 5. 离线测试

![完整离线回放流程](svg/06-offline-test.svg)

### 完整序列流程

1. 从开发验证集或独立留出集中取出一条完整实机记录。每条记录包含 3、4 或 5 铲，并带有每一铲的左右目标顺序。
2. 开始一铲前，脚本规划器只提交这一铲的目标。模型会清空上一铲留下的动作预测和内部历史，避免旧目标影响新一铲的第一步。
3. 按原始 20 Hz 顺序逐帧读取四路图像、关节位置和关节速度。模型给出动作后，再经过与运行时一致的动作保护程序。每一帧同时保存模型原始动作、保护后的动作和专家动作。
4. 根据记录中的真实回转位置划分阶段：到回转最高点为挖掘与卸料；从最高点回到目标区为回程；进入目标区直到连续十帧低速度稳定为释放和回位。
5. 回程经过右区决策范围时，追加一次“同一个状态只换左右目标”的对照。专家动作刚从死区内变成有效动作时，记录一个“机器没动时重新判断”的测试点。
6. 记录数据确认目标区稳定回位后，当前一铲结束，规划器提交下一铲目标。重复以上步骤，直到整条目标序列结束。
7. 最后分别汇总开发验证集和独立留出集。完整动作、左右目标和机器不动三类指标必须一起通过。

周期推进以记录中的稳定回位边界为准。

### 左右目标对照

每一铲经过回转最高点、进入训练数据支持的右区范围后，代码最多均匀取十六个时刻。每个时刻的四路图像、关节位置和关节速度完全相同，只把目标分别改成左区和右区。

为了避免前几步动作平均影响结果，代码另开一个不使用历史动作平均的模型副本，并在每次判断前清空内部记录。比较的是模型在当前这一步直接给出的命令。左区命令必须达到负向机械有效幅值，右区命令必须留在死区内。两项在同一个时刻同时正确，才算一次成对命中。

### 机器不动测试

![机器不动测试](svg/06-state-hold.svg)

1. 在专家动作刚从死区内变为机械有效动作的位置选一个测试点，并剔除原数据中不可训练或不完整的时刻。
2. 从这段数据开头逐帧运行到测试点，还原模型当时对过去几步的内部记录。
3. 保存模型此刻的内部状态。每次测试一个轴和方向前，都恢复到这份完全相同的状态。
4. 保持图像、关节位置和左右目标不变，把关节速度设为零，然后让模型在同一个状态下连续判断二十次。模型动作不会让状态前进，也看不到专家随后已经让机器动起来的下一帧。
5. 检查二十次以内是否重新给出了同一轴、同一方向、且越过死区的动作；同时记录第一次判断是否给了反方向或多余的其他轴动作。
6. 测试结束后恢复模型状态，再继续读取真实记录。这样一个测试点不会影响下一个测试点。

每段数据的第一个测试点单独统计，用来判断模型能否从静止开始；其余测试点检查动作过程中能否继续给出有效动作。前五次只观察反应快慢，最终验收看二十次以内能否重新给出有效动作。

### 状态数据来源

动作积分方案在专家动作回放中产生了数弧度的位置误差，也无法同步生成对应相机画面。离线测试因此使用记录中的真实图像、位置和速度。

这部分评价模型在已记录状态下给出的命令，以及机器不动时的重复决策。

### 验收标准

开发验证集和独立留出集分别要求：动作误差不超过 0.13，方向一致率不低于 90%，左右目标成对命中不低于 80%，右区停止不低于 95%；挖掘和回程表现至少达到对应专家数据的 80%。同时要求所有动作数值正常、策略运行无报错、动作保护不被触发。

机器不动测试在相同数据、测试点和死区表上对比原一铲模型。新模型的同向有效动作比例不低于原模型，错误方向或多余动作次数不高于原模型。

对应代码为：[完整序列和左右目标测试](../../testbed/testbed/cli/planner_open_loop_replay.py)、[机器不动测试](../../testbed/testbed/cli/state_hold_transition_replay.py)和[汇总验收](../../testbed/testbed/cli/real_transition_acceptance.py)。

## 6. 结果

![左右目标响应与死区能力](svg/01-conclusion.svg)

### 左右目标响应

开发验证集的左右成对命中率为 99.0%，独立留出集为 92.7%；两套数据中，右区目标的回零率都是 100%。这说明同一个状态只换左右目标时，模型能够分别给出继续回转和停止回转两种机械命令。

### 原有能力保持

![离线结果](svg/07-results.svg)

普通动作方向一致率在两套数据中分别为 92.9% 和 91.6%。挖掘、回程、动作误差、策略运行和动作保护也全部通过预定标准。

机器保持不动时，二十次判断内重新给出同向有效动作的比例，在开发验证集由原模型的 55.4% 提高到 67.5%，在独立留出集由 58.4% 保持到 59.7%。第一次就给错方向或多余动作的测试点数也从 38 降到 26、从 53 降到 29。

完整序列回放覆盖左到左、左到右、右到左、右到右四种转换。

### 结论

新的回转释放损失增加了左右目标响应，同时保留原一铲模型的普通动作和死区能力。

当前结论来自离线动作测试；液压响应、实际目标区域和连续多铲运行由真机测试确认。
"""


def main() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    build_work_map()
    build_conclusion()
    build_data_distribution()
    build_failed_designs()
    build_method()
    build_theory()
    build_offline_test()
    build_state_hold_test()
    build_results()
    MD_PATH.write_text(MARKDOWN, encoding="utf-8")
    print(
        {
            "markdown": str(MD_PATH),
            "svg_dir": str(SVG_DIR),
            "svg_count": len(list(SVG_DIR.glob("*.svg"))),
        }
    )


if __name__ == "__main__":
    main()
