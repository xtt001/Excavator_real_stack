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
    body.append(t(78, 195, "本周只检查两件事：模型是否听从左右目标，模型原来的表现有没有变差。", size=27, fill=MUTED))
    tasks = [
        (70, TEAL, "让模型听从左右目标", ("目标在左区时，模型继续回转", "目标在右区时，模型停下")),
        (570, ORANGE, "保持原有表现", ("模型照常完成挖掘、卸料和回程", "机器卡住时，模型仍能再次发力")),
        (1070, BLUE, "两项一起过关", ("开发验证集和独立留出集", "都要同时满足两项要求")),
    ]
    for x, colour, title, detail in tasks:
        body.append(rect(x, 230, 460, 156, fill=WHITE))
        body.append(f'<rect x="{x}" y="230" width="9" height="156" fill="{colour}"/>')
        body.append(t(x + 32, 276, title, size=27, fill=colour, weight=700))
        body.append(lines(x + 32, 324, detail, size=22, line_height=36, fill=INK, weight=500))
    body.append(t(78, 450, "本周做了六步", size=27, fill=NAVY, weight=700))
    stages = [
        (70, BLUE, "1", "先看训练数据", ("找出左右动作", "真正不同的时刻")),
        (310, RED, "2", "试三种办法", ("直接检查动作", "有没有改变")),
        (550, TEAL, "3", "定下训练办法", ("确定取样时刻", "动作要求和权重")),
        (790, PURPLE, "4", "训练并选择模型", ("按离线动作表现选", "不只看损失")),
        (1030, ORANGE, "5", "跑离线测试", ("跑完整记录并检查", "左右目标和卡住情况")),
        (1270, NAVY_2, "6", "给出结论", ("说明已经做到什么", "还要上机确认什么")),
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
    body.append(t(800, 732, "从训练到离线测试，这两项要求始终一起检查。", size=24, fill=WHITE, weight=700, anchor="middle"))
    save_svg(
        "00-work-map.svg",
        document(
            title="本周任务与时间线",
            description="本周先从训练数据中找出左右目标真正需要不同动作的时刻，再比较三种训练办法、训练模型并完成离线测试。",
            height=790,
            body=body,
        ),
    )


def build_conclusion() -> None:
    body: list[str] = []
    body.append(t(78, 198, "离线测试表明，模型已经听从左右目标，也能像以前一样完成整铲动作。", size=27, fill=MUTED))
    cards = [
        (78, TEAL, "99.0%", "开发验证集", "左右动作同时正确"),
        (579, BLUE, "92.7%", "独立留出集", "左右动作同时正确"),
        (1080, ORANGE, "不低于原模型", "两套数据", "机器卡住时再次发力"),
    ]
    for x, colour, value, label, detail in cards:
        body.append(rect(x, 235, 442, 222, fill=WHITE))
        body.append(f'<rect x="{x}" y="235" width="10" height="222" fill="{colour}"/>')
        body.append(t(x + 221, 315, value, size=50 if x < 1000 else 41, fill=colour, weight=700, anchor="middle"))
        body.append(t(x + 221, 370, label, size=24, fill=MUTED, weight=700, anchor="middle"))
        body.append(t(x + 221, 412, detail, size=25, fill=INK, anchor="middle"))
    body.append(rect(78, 500, 1444, 225, fill=WHITE))
    body.append(t(115, 548, "模型具体要做到什么", size=29, fill=NAVY, weight=700))
    rows = [
        (595, TEAL, "目标在左区", "模型继续负向回转，而且命令足以让机器转动"),
        (652, BLUE, "目标在右区", "模型把回转命令降到零，让机器停在右区"),
        (709, ORANGE, "机器卡住", "即使机器没动，模型也能再次给出越过死区的命令"),
    ]
    for y, colour, label, detail in rows:
        body.append(pill(115, y - 35, 154, label, fill=colour, height=46, size=21))
        body.append(t(300, y, detail, size=26, fill=INK, weight=600))
    save_svg(
        "01-conclusion.svg",
        document(
            title="模型听从左右目标，也保持了原有表现",
            description="新模型在两套离线数据上都能按照左右目标给出不同动作，机器卡在死区时的表现也不低于原模型。",
            height=780,
            body=body,
        ),
    )


def build_data_distribution() -> None:
    body: list[str] = []
    body.append(t(78, 198, "一整铲里，模型在左右目标下的大多数动作都相同，只有停下前的一小段不同。", size=27, fill=MUTED))
    body.append(t(95, 255, "一整铲的动作顺序", size=27, fill=NAVY, weight=700))
    segments = [
        (95, 180, "挖掘", "#B9C7D5"),
        (285, 240, "抬升与装载", "#9EB1C4"),
        (535, 300, "正向回转卸料", "#7894AD"),
        (845, 350, "共同负向回转", "#4D789D"),
        (1205, 300, "继续回转或停下", ORANGE),
    ]
    for x, width, label, colour in segments:
        body.append(rect(x, 290, width, 82, fill=colour, stroke=colour, radius=16))
        body.append(t(x + width / 2, 342, label, size=22, fill=WHITE, weight=700, anchor="middle"))
    body.append(t(95, 438, "训练时间步分布", size=27, fill=NAVY, weight=700))
    bar_x, bar_y, bar_width = 95, 470, 1410
    common_width = int(bar_width * 0.9415)
    body.append(rect(bar_x, bar_y, bar_width, 78, fill="#DCE5ED", stroke="#DCE5ED", radius=18))
    body.append(rect(bar_x + common_width, bar_y, bar_width - common_width, 78, fill=ORANGE, stroke=ORANGE, radius=18))
    body.append(t(bar_x + common_width / 2, bar_y + 50, "两种目标下相同的动作 94.15%", size=27, fill=NAVY, weight=700, anchor="middle"))
    body.append(line(bar_x + common_width, bar_y - 10, bar_x + common_width, bar_y + 90, stroke=WHITE, width=4))
    body.append(t(1505, 590, "目标可能影响动作的区间 5.85%", size=25, fill=ORANGE, weight=700, anchor="end"))
    body.append(line(1432, 554, 1464, 574, stroke=ORANGE, width=3))
    body.append(rect(95, 618, 1410, 92, fill=WHITE))
    body.append(t(130, 674, "真正能比较左右动作的起始时刻只有 1.86%", size=27, fill=RED, weight=700))
    body.append(t(770, 674, "如果整段平均训练，这些少量时刻很容易被大量相同动作盖住。", size=26, fill=INK, weight=600))
    save_svg(
        "02-data-distribution.svg",
        document(
            title="左右目标只在回转停止前产生差别",
            description="左右目标在一整铲中有百分之九十四以上的动作相同，真正的差别集中在回程经过右区、需要决定继续回转还是停下的时刻。",
            height=760,
            body=body,
        ),
    )


def build_failed_designs() -> None:
    body: list[str] = []
    body.append(t(78, 198, "三种办法的问题不同：一种只学会分类，一种基于错误假设，一种把动作卡在死区边缘。", size=27, fill=MUTED))
    columns = [
        (150, "怎么做"),
        (575, "希望达到什么效果"),
        (850, "实际结果"),
        (1320, "问题"),
    ]
    body.append(rect(70, 225, 1460, 58, fill=NAVY, stroke=NAVY, radius=14))
    for x, label in columns:
        body.append(t(x, 263, label, size=22, fill=WHITE, weight=700))
    rows = [
        (
            298,
            RED,
            "1",
            ("让模型顺便判断左区或右区", "测试权重 0.5、1、2、5"),
            ("先认出目标，再让", "动作跟着改变"),
            ("权重 1、2、5 时，分类准确率达到 100%", "回转动作差最大仅 0.014–0.016"),
            ("会分类，但动作差", "小到无法让机器转动"),
        ),
        (
            460,
            ORANGE,
            "2",
            ("目标在左区时负向回转", "目标在右区时正向回转"),
            ("用相反方向", "强行拉开动作差别"),
            ("两种目标下，人工驾驶都主要使用负向回转", "区别是何时停止，不是方向相反"),
            ("这个假设不符合", "真实动作"),
        ),
        (
            622,
            PURPLE,
            "3",
            ("目标在左区：动作 ≤ −0.721", "目标在右区：动作 = 0"),
            ("直接规定左区继续转", "右区停下"),
            ("损失为零时，动作仍可卡在死区边缘", "短训左右同时正确约 67%–74%"),
            ("动作没有余量，而且", "没有达到 80% 标准"),
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
    body.append(t(275, 847, "只在需要选择继续或停止的时刻加要求：左区 ≤ −0.799，右区 = 0，两者至少相差 0.799。", size=23, fill=INK, weight=700))
    save_svg(
        "03-failed-designs.svg",
        document(
            title="试过的三种训练办法",
            description="第一种办法只让模型学会左右分类，第二种办法错误地假设左右动作方向相反，第三种办法允许动作停在死区边缘。",
            height=920,
            body=body,
        ),
    )


def build_method() -> None:
    body: list[str] = []
    nodes = [
        (85, 220, 290, "图像和机器状态", PALE_BLUE, BLUE),
        (445, 220, 270, "挑出需要选择左右的时刻", PALE_ORANGE, ORANGE),
        (785, 220, 280, "分别给出左区和右区目标", PALE_TEAL, TEAL),
        (1135, 220, 380, "模型预测接下来的一段动作", PALE_PURPLE, PURPLE),
    ]
    for x, y, width, label, background, colour in nodes:
        body.append(rect(x, y, width, 98, fill=background, stroke=background))
        body.append(t(x + width / 2, y + 61, label, size=25, fill=colour, weight=700, anchor="middle"))
    for x1, x2 in ((375, 445), (715, 785), (1065, 1135)):
        body.append(line(x1 + 10, 269, x2 - 12, 269, stroke="#A9B9C8", width=5, arrow=True))
    body.append(t(85, 382, "新增的左右回转损失", size=29, fill=NAVY, weight=700))
    body.append(rect(85, 410, 1430, 108, fill=NAVY, stroke=NAVY, radius=22))
    formula = "L_release = max(0, m + a_left) + |a_right| + max(0, m − (a_right − a_left))"
    body.append(t(800, 478, formula, size=31, fill=WHITE, weight=700, anchor="middle"))
    implications = [
        (85, TEAL, "目标在左区", "a_left ≤ −m", "继续回转，命令要越过死区"),
        (565, BLUE, "目标在右区", "a_right = 0", "把回转命令降到零并停下"),
        (1045, ORANGE, "左右动作差", "a_right − a_left ≥ m", "避免预测误差把动作推回死区"),
    ]
    for x, colour, label, value, detail in implications:
        body.append(rect(x, 560, 430, 132, fill=WHITE))
        body.append(pill(x + 28, 584, 138, label, fill=colour, height=44, size=20))
        body.append(t(x + 195, 616, value, size=25, fill=colour, weight=700))
        body.append(t(x + 28, 665, detail, size=23, fill=INK, weight=600))
    parameters = [
        ("右区位置范围", "0.111–0.393 弧度"),
        ("负向回转死区", "0.721"),
        ("左区目标幅值 m", "0.799"),
        ("连续检查", "5 帧 / 0.25 秒"),
        ("新增损失权重", "0.503"),
    ]
    card_width = 274
    for index, (label, value) in enumerate(parameters):
        x = 85 + index * 286
        body.append(rect(x, 728, card_width, 96, fill=WHITE))
        body.append(t(x + 20, 764, label, size=20, fill=MUTED, weight=700))
        body.append(t(x + 20, 805, value, size=23, fill=BLUE, weight=700))
    body.append(t(800, 870, "其他时刻的训练仍使用原来的样本和损失。新增损失只检查模型应该继续回转还是停下。", size=24, fill=RED, weight=700, anchor="middle"))
    save_svg(
        "04-method.svg",
        document(
            title="最终采用的训练办法",
            description="我们只在需要选择继续回转还是停下的时刻增加损失。新增损失直接要求模型在目标为左区时继续回转，在目标为右区时停止，并让两种动作留出足够差距。",
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
        "长序列中，真正需要改变动作的时刻很少",
        "如果每一帧同样训练，模型容易照抄前一个动作",
        "让模型多练这些关键时刻，可以改善模仿效果",
    ]
    for index, value in enumerate(findings):
        body.append(bullet(115, 390 + index * 88, value, colour=BLUE, size=24))
    body.append(rect(830, 200, 700, 410, fill=WHITE))
    body.append(t(870, 255, "与本项目的对应", size=31, fill=NAVY, weight=700))
    mappings = [
        ("需要换动作的时刻很少", "回到右区后，模型才决定继续还是停下"),
        ("大多数动作都相同", "挖掘、装载和卸料时，模型给出相同动作"),
        ("多练关键时刻", "每铲多取一个目标开始影响动作的样本"),
        ("模型容易照抄旧动作", "模型忽略左右目标，最后给出折中的动作"),
    ]
    for index, (left, right) in enumerate(mappings):
        y = 335 + index * 64
        body.append(t(875, y, left, size=23, fill=MUTED))
        body.append(t(1135, y, "→", size=25, fill=ORANGE, weight=700, anchor="middle"))
        body.append(t(1160, y, right, size=20, fill=INK, weight=700))
    body.append(t(80, 665, "其他相关研究", size=27, fill=NAVY, weight=700))
    references = [
        (80, BLUE, "把任务拆成阶段", "TACO · CompILE", "把长任务拆成几个动作阶段"),
        (460, TEAL, "按实际结果补目标", "HER · GCSL", "用已经发生的结果补充目标样本"),
        (840, ORANGE, "用目标控制动作", "Conditional Imitation", "同一模型按高层目标做不同动作"),
        (1220, PURPLE, "真机数据", "本项目", "这些数据决定动作大小和死区"),
    ]
    for x, colour, title, ref, detail in references:
        body.append(rect(x, 700, 300, 112, fill=WHITE))
        body.append(f'<rect x="{x}" y="700" width="8" height="112" fill="{colour}"/>')
        body.append(t(x + 28, 738, title, size=23, fill=colour, weight=700))
        body.append(t(x + 28, 770, ref, size=19, fill=MUTED, weight=700))
        body.append(t(x + 28, 798, detail, size=18, fill=INK))
    body.append(t(800, 866, "论文说明为什么要多训练少数关键时刻。真机数据决定具体位置、动作大小和死区。", size=24, fill=RED, weight=700, anchor="middle"))
    save_svg(
        "05-theory.svg",
        document(
            title="为什么要盯住少数关键时刻",
            description="关键帧模仿学习说明，长序列中真正需要改变动作的时刻很少，给这些时刻更多训练机会可以减少模型照抄旧动作的问题。",
            height=900,
            body=body,
        ),
    )


def build_offline_test() -> None:
    body: list[str] = []
    body.append(rect(78, 182, 1444, 74, fill=PALE_BLUE, stroke=PALE_BLUE, radius=18))
    body.append(pill(105, 195, 140, "测试单位", fill=BLUE, height=46, size=20))
    body.append(t(270, 228, "一条完整实机记录，里面有 3、4 或 5 铲。测试从第一个目标一直跑到最后一个目标。", size=24, fill=INK, weight=600))
    body.append(t(78, 304, "一条完整记录怎样跑完", size=27, fill=NAVY, weight=700))
    nodes = [
        (70, 190, "读入整段记录", "按照实机顺序逐铲读取"),
        (280, 190, "给出这一铲目标", "每铲开始只给一个目标"),
        (490, 210, "清空上一铲历史", "旧动作预测和内部记录归零"),
        (720, 210, "按 20 Hz 读取数据", "四路图像、位置和速度"),
        (950, 170, "模型给动作", "保存模型原始输出"),
        (1140, 170, "通过安全保护", "保存保护后的动作"),
        (1330, 200, "分阶段统计", "挖掘→回程→停下→稳定"),
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
            "① 每一帧都检查",
            [
                "代码保存模型输出、保护后动作和人工驾驶动作",
                "代码按最高点、目标区入口和停稳时刻分段",
                "代码统计动作误差和方向，并检查各阶段表现",
            ],
        ),
        (
            565,
            TEAL,
            "② 回到右区时比较左右目标",
            [
                "代码只在训练数据覆盖的右区范围内取点",
                "模型在相同图像和状态下分别接收左右目标",
                "测试关闭历史平均，只看模型当前给出的动作",
            ],
        ),
        (
            1060,
            ORANGE,
            "③ 机器刚要动起来时检查",
            [
                "代码在人工驾驶动作刚越过死区时选点",
                "测试固定状态，让模型连续判断二十次",
                "测试记录模型何时给对动作，以及是否给错动作",
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
    body.append(t(800, 826, "记录显示机器已经稳定回位后，这一铲才结束，规划器随后给出下一铲目标。", size=24, fill=WHITE, weight=700, anchor="middle"))
    body.append(t(800, 859, "代码分别统计两套数据。整段动作是否准确、模型是否听从目标、机器卡住时能否再次发力，这三项都要通过。", size=20, fill="#D5E4EF", anchor="middle"))
    save_svg(
        "06-offline-test.svg",
        document(
            title="离线测试怎样跑完一条记录",
            description="测试每次读取一条包含三到五铲的实机记录，规划器逐铲给出目标，模型按照真实图像、位置和速度给出动作，并在记录显示机器稳定回位后进入下一铲。",
            height=930,
            body=body,
        ),
    )


def build_state_hold_test() -> None:
    body: list[str] = []
    body.append(t(78, 198, "如果命令没有越过死区，机器就不会动，模型下一次看到的还是同一个状态。这个测试复现的就是这种情况。", size=25, fill=MUTED))
    steps = [
        (
            70,
            BLUE,
            "1. 找出机器刚要动的时刻",
            ("人工驾驶的回转命令刚越过死区", "代码按轴和方向标记测试点", "并排除无效样本"),
        ),
        (
            455,
            PURPLE,
            "2. 重放此前记录",
            ("代码逐帧读取测试点前的数据", "模型因此重新看到", "此前的图像和机器状态"),
        ),
        (
            840,
            ORANGE,
            "3. 每次回到同一状态",
            ("代码保存模型的内部记录", "每次测试前都恢复这份记录", "图像、位置和目标不变，速度为零"),
        ),
        (
            1225,
            TEAL,
            "4. 让模型连续判断",
            ("让模型对同一份数据判断二十次", "测试不更新机器状态", "也不读取人工驾驶的下一帧"),
        ),
    ]
    for x, colour, title, detail in steps:
        body.append(rect(x, 235, 315, 168, fill=WHITE))
        body.append(rect(x, 235, 315, 54, fill=colour, stroke=colour, radius=20))
        body.append(t(x + 157.5, 272, title, size=24, fill=WHITE, weight=700, anchor="middle"))
        body.append(lines(x + 24, 322, detail, size=17, line_height=27, fill=INK, weight=500))
    for x1, x2 in ((385, 455), (770, 840), (1155, 1225)):
        body.append(line(x1 + 10, 310, x2 - 12, 310, stroke="#9FB1C2", width=4, arrow=True))
    body.append(t(80, 446, "怎样保证每次测试公平", size=27, fill=NAVY, weight=700))
    fairness = [
        (80, BLUE, "1", "先重放真实记录", "代码逐帧运行到测试点"),
        (600, ORANGE, "2", "保存模型此时的记录", "每项测试前都恢复这份记录"),
        (1120, TEAL, "3", "逐项测试后继续回放", "测试结束后恢复记录，再读取后续数据"),
    ]
    for x, colour, number, title, detail in fairness:
        body.append(rect(x, 474, 400, 118, fill=WHITE))
        body.append(circle(x + 38, 533, 23, fill=colour))
        body.append(t(x + 38, 541, number, size=22, fill=WHITE, weight=700, anchor="middle"))
        body.append(t(x + 78, 519, title, size=21, fill=colour, weight=700))
        body.append(t(x + 78, 556, detail, size=18, fill=INK))
    body.append(line(490, 533, 585, 533, stroke="#9FB1C2", width=4, arrow=True))
    body.append(line(1010, 533, 1105, 533, stroke="#9FB1C2", width=4, arrow=True))
    body.append(t(80, 650, "测试检查什么", size=27, fill=NAVY, weight=700))
    criteria = [
        (80, BLUE, "5 次内给对动作", "只用来观察反应快慢，不作为最终门槛"),
        (450, TEAL, "20 次内给对动作", "机器没有动时，能否再次发出有效命令"),
        (820, RED, "第一次就给错动作", "目标轴方向相反，或其他轴出现多余动作"),
        (1190, ORANGE, "区分启动和动作中", "第一个测试点单独统计，避免平均数盖住启动问题"),
    ]
    for x, colour, title, detail in criteria:
        body.append(rect(x, 680, 330, 102, fill=WHITE))
        body.append(f'<rect x="{x}" y="680" width="7" height="102" fill="{colour}"/>')
        body.append(t(x + 24, 720, title, size=21, fill=colour, weight=700))
        body.append(t(x + 24, 755, detail, size=17, fill=INK))
    body.append(rect(80, 820, 1440, 76, fill=NAVY, stroke=NAVY, radius=18))
    body.append(t(120, 851, "新旧模型使用同一批数据、同一批测试点和同一张死区表：", size=22, fill=WHITE, weight=700))
    body.append(t(900, 851, "开发 67.5% ≥ 55.4%，错误 26 ≤ 38", size=21, fill="#76E0D1", weight=700))
    body.append(t(900, 880, "留出 59.7% ≥ 58.4%，错误 29 ≤ 53", size=21, fill="#76E0D1", weight=700))
    save_svg(
        "06-state-hold.svg",
        document(
            title="机器卡住时，模型会不会再次发力",
            description="测试先重放真实记录来还原模型到达测试点时的状态，然后固定图像、位置和目标，把速度设为零，让模型连续判断二十次，并与原模型比较。",
            height=940,
            body=body,
        ),
    )


def build_results() -> None:
    body: list[str] = []
    metrics = [
        (70, 350, TEAL, "99.0%", "开发验证集", "左右动作同时正确"),
        (450, 350, BLUE, "92.7%", "独立留出集", "左右动作同时正确"),
        (830, 350, ORANGE, "100%", "两套数据", "回转命令降到零"),
        (1210, 320, NAVY_2, "0", "两套数据", "报错或保护触发次数"),
    ]
    for x, width, colour, value, label, detail in metrics:
        body.append(rect(x, 195, width, 190, fill=WHITE))
        body.append(f'<rect x="{x}" y="195" width="9" height="190" fill="{colour}"/>')
        body.append(t(x + width / 2, 270, value, size=51, fill=colour, weight=700, anchor="middle"))
        body.append(t(x + width / 2, 322, label, size=22, fill=MUTED, weight=700, anchor="middle"))
        body.append(t(x + width / 2, 356, detail, size=22, fill=INK, anchor="middle"))
    body.append(rect(70, 425, 720, 405, fill=WHITE))
    body.append(t(105, 474, "模型完成一铲的表现", size=29, fill=NAVY, weight=700))
    headers = [(120, "指标"), (430, "开发验证集"), (635, "独立留出集")]
    body.append(f'<rect x="105" y="505" width="650" height="56" fill="{PALE_BLUE}"/>')
    for x, value in headers:
        body.append(t(x, 542, value, size=21, fill=NAVY, weight=700))
    table_rows = [
        ("动作方向一致率", "92.9%", "91.6%"),
        ("挖掘阶段有效率", "23.4%", "24.7%"),
        ("回转动作有效率", "66.8%", "60.9%"),
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
    body.append(t(865, 474, "机器没动时，模型能否在 20 次内再次给对动作", size=24, fill=NAVY, weight=700))
    groups = [
        ("开发验证集", 55.4, 67.5, 38, 26, 550),
        ("独立留出集", 58.4, 59.7, 53, 29, 690),
    ]
    for label, baseline, candidate, old_wrong, new_wrong, y in groups:
        body.append(t(865, y - 34, label, size=22, fill=MUTED, weight=700))
        body.append(t(1120, y - 34, f"模型第一次就给错方向或产生多余动作：{old_wrong} → {new_wrong}", size=16, fill=RED, weight=700))
        scale = 560 / 80.0
        body.append(rect(865, y, baseline * scale, 34, fill="#B6C7D8", stroke="#B6C7D8", radius=17))
        body.append(t(875 + baseline * scale, y + 25, f"原模型 {baseline:.1f}%", size=19, fill=NAVY, weight=700))
        body.append(rect(865, y + 48, candidate * scale, 34, fill=TEAL, stroke=TEAL, radius=17))
        body.append(t(875 + candidate * scale, y + 73, f"新模型 {candidate:.1f}%", size=19, fill=TEAL, weight=700))
    body.append(t(800, 875, "模型已经听从左右目标，而且机器卡住时再次发力的表现不低于原模型。", size=27, fill=TEAL, weight=700, anchor="middle"))
    save_svg(
        "07-results.svg",
        document(
            title="离线测试结果",
            description="在相同状态下分别输入左右目标时，两种动作同时正确的比例达到百分之九十九和百分之九十二点七；机器卡住时再次给出有效命令的比例也不低于原模型。",
            height=920,
            body=body,
        ),
    )


MARKDOWN = """# 让模型按照左右目标完成一铲

**本周组会 · 2026-08-28**

> 新模型能够按照脚本给出的左区或右区目标，选择继续回转或在右区停下。它完成挖掘、卸料和回程的表现没有变差。机器卡在死区时，它仍会再次给出有效命令。

![本周任务与时间线](svg/00-work-map.svg)

## 本周要解决的两件事

第一件事是让模型真正听从脚本给出的左右目标。机器回程经过右区时，如果目标在左区，模型就要继续负向回转。如果目标在右区，模型就要把回转命令降到零，让机器停在右区。

第二件事是防止模型原来的表现变差。新增训练不能降低模型在挖掘、卸料和回程中的表现。机器卡住时，模型仍要再次给出有效命令。

只有这两项检查都通过，模型才有资格进入真机测试。

## 1. 找出左右目标真正需要不同动作的时刻

![左右目标只在回转停止前产生差别](svg/02-data-distribution.svg)

训练集有 40,595 个时间步，覆盖 60 个完整周期。对于其中 94.15% 的时间步，目标在左区还是右区并不会改变人工驾驶动作。目标可能影响动作的区间只占 5.85%，真正能直接比较左右动作的起始时刻只有 1.86%。

在挖掘、装载、卸料和大部分回程过程中，模型面对两种目标时本来就应该给出相同动作。只有回程经过右区时，模型才需要决定接下来怎么做。目标在左区就继续回转，目标在右区就在当前位置停下。

负向回转命令的幅值不足 0.721 时，真机通常不会转动，这段无效范围就是回转死区。

离线测试需要回答两个问题：

1. 同一幅画面和同一组机器状态下，只改变左右目标，模型会不会改变回转命令？
2. 加入左右目标训练以后，模型完成一铲和越过死区时的表现有没有变差？

## 2. 比较三种训练办法

![试过的三种训练办法](svg/03-failed-designs.svg)

### 办法一　让模型顺便判断左区还是右区

第一种办法在动作预测之外增加了一个左右分类任务，并测试了 0.5、1、2、5 四个损失权重。我们希望模型先认出目标，再让动作跟着目标改变。

当权重为 1、2、5 时，左右分类准确率都达到了 100%。但是，同一个状态只换目标后，回转动作最多只相差 0.014–0.016，远小于让真机开始回转所需的 0.721。模型学会了分类，却没有给出足以让真机作出不同反应的回转命令。

### 办法二　直接规定左右动作相反

第二种办法要求模型在目标为左区时负向回转，在目标为右区时正向回转。这样可以用相反方向强行拉开两种动作。

真机数据不支持这个假设。在两种目标下，人工驾驶的回程动作都以负向回转为主。目标在右区时，司机只是更早把负向回转命令松到零，并不会改成正向回转。

### 办法三　按死区边界规定动作

第三种办法要求，目标在左区时，回转动作不高于 −0.721。目标在右区时，回转动作回到 0。两种动作还要至少相差 0.721。

只要左区动作刚好达到 −0.721，新增损失就可能降到零。这样的动作正好卡在死区边缘，模型稍有误差就无法让机器转动。三组 80 轮短训中，同一状态下左区继续转、右区停下且两项都正确的比例分别为 67.0%、67.8% 和 74.3%，没有达到预定的 80%。

### 最后采用的改法

最终训练要求左区动作的幅值至少达到 0.799，也就是动作不高于 −0.799。0.799 是训练数据中有效回转动作的中位数，所以这个数来自真实操作数据，并且比 0.721 的死区边界留出了余量。右区动作仍然要求回到 0，左右动作至少相差 0.799。

## 3. 最终怎么训练

![最终采用的训练办法](svg/04-method.svg)

### 每铲只挑一个关键时刻

当回转位置越过最高点、机器回到右区，而且人工驾驶动作明确表示继续或停止时，我们才增加训练样本。每铲只额外取一个这样的时刻，防止某一铲因为可选时刻更多而在训练中占得过重。

### 损失直接检查模型给出的动作

训练程序会把同一份图像和机器状态输入模型两次，只有左右目标不同。新增损失直接检查三件事：

1. 目标在左区时，模型接下来五步的回转动作都要达到 −0.799 或更负。
2. 目标在右区时，模型接下来五步的回转动作都要回到 0。
3. 两组动作至少要相差 0.799，避免预测误差把左区动作推回死区。

### 这些数值来自哪里

右区的位置范围、0.721 的负向回转死区、0.799 的左区动作要求，以及连续检查五步的做法都来自训练数据。连续检查五步对应 0.25 秒，可以排除只在一帧上偶然出现的动作。

新增损失的权重是 0.503。这个数由训练开始时原有损失和新增损失的大小换算得到，使两部分在训练初期处在相近的影响范围。

### 原来的训练照常进行

原有的模仿训练、死区训练，以及“机器不动时再次给动作”的样本仍然保留。新增损失只处理回程经过右区时应该继续回转还是停下，不会重新规定整铲的其他动作。

### 为什么要集中训练这些时刻

![为什么要盯住少数关键时刻](svg/05-theory.svg)

[Keyframe-Focused Visual Imitation Learning](https://proceedings.mlr.press/v139/wen21d.html) 研究了长动作序列中的少数关键时刻。如果每一帧得到的训练机会一样，大量重复动作很容易盖住少数真正需要改变命令的时刻。每一铲中，我们都会额外选取一个左右目标开始产生不同动作的时刻。这沿用了论文提出的思路。

[TACO](https://proceedings.mlr.press/v80/shiarlis18a.html) 和 [CompILE](https://proceedings.mlr.press/v97/kipf19a.html) 表明，长任务可以按动作阶段拆开学习。这些研究解释了为什么要单独处理回转停止前的选择。具体的位置范围、动作大小和死区仍然由真机数据决定。

## 4. 训练模型并选择版本

训练共运行 300 轮。第 140 轮的验证损失最低，但验证损失是整段动作的平均结果，不能单独说明模型是否听从左右目标。我们因此保存了多个训练轮次的模型，并逐个运行离线动作测试。

第 199 轮模型通过了三类检查：它会按照左右目标给动作，完成一铲的表现没有变差，而且在机器卡住时也能再次发力。因此，我们选择这个版本准备上机测试。

## 5. 离线测试怎么做

![离线测试怎样跑完一条记录](svg/06-offline-test.svg)

### 先跑完一条完整记录

1. 测试先从开发验证集或独立留出集中读取一条完整实机记录。每条记录包含 3、4 或 5 铲，并保存了每一铲的左右目标。
2. 每铲开始前，脚本规划器只给出这一铲的目标。测试同时清空模型在上一铲留下的动作预测和内部记录，防止旧目标影响新一铲。
3. 测试按照原始 20 Hz 的顺序读取四路图像、关节位置和关节速度。模型给出动作以后，动作还要经过与真机运行时相同的安全保护。代码会同时保存模型动作、保护后的动作和人工驾驶动作。
4. 代码根据记录中的真实回转位置划分动作阶段。代码把回转位置达到最高点以前的部分记为挖掘和卸料阶段，把从最高点返回目标区的部分记为回程阶段。进入目标区以后，速度还要连续十帧保持较低，代码才会确认机器已经停稳。
5. 机器回程经过右区时，测试会用同一个状态分别输入左右目标。人工驾驶动作刚越过死区时，测试还会记录一个“机器没动时再次判断”的检查点。
6. 只有记录显示机器已经在目标区稳定停下，当前一铲才算结束。规划器随后给出下一铲目标，并一直运行到整条目标序列结束。
7. 代码分别统计开发验证集和独立留出集。测试会检查整段动作是否准确、模型是否听从左右目标，以及机器卡住时能否再次发力。这三类检查必须一起通过。

### 在同一状态下比较左右目标

每一铲中，回转位置经过最高点并进入训练数据覆盖的右区范围后，代码最多均匀选取十六个时刻。对于每个时刻，测试都使用完全相同的四路图像、关节位置和关节速度，只把目标分别改成左区和右区。

ACT 正常运行时会把前几个时刻留下的动作预测一起平均，这可能会冲淡当前目标造成的动作差别。为了直接检查目标有没有改变模型这一步的输出，测试会关闭历史动作平均，并在每次判断前清空内部记录。目标在左区时，命令必须足以让机器负向回转。目标在右区时，命令必须留在死区内。只有两项在同一个时刻都正确，这次判断才算通过。

### 机器卡住时，检查模型会不会再次发力

![机器卡住时的测试过程](svg/06-state-hold.svg)

1. 测试先找到人工驾驶的回转命令刚从死区内跨到死区外的时刻，并排除数据不完整或无法训练的时刻。
2. 代码从这段记录的开头逐帧运行到测试点，让模型重新看到此前的图像和机器状态。
3. 到达测试点后，代码保存模型的内部记录。每次测试一个轴和一个方向之前，代码都会把模型恢复到同一份记录。
4. 测试保持图像、关节位置和左右目标不变，把关节速度设为零，然后让模型连续判断二十次。测试不会根据模型动作更新图像、位置或速度，模型也看不到人工驾驶随后已经让机器动起来的下一帧。
5. 测试检查模型能否在二十次以内，再次为同一个轴给出方向相同且足以越过死区的命令。同时，代码会记录模型第一次判断时是否给错方向，或者是否在其他轴上给出了多余动作。
6. 一个测试点结束后，代码会恢复此前保存的状态，再继续读取真实记录。因此，前一个测试点不会改变后一个测试点的输入。

每段记录的第一个测试点单独统计，用来检查模型能否从静止状态开始发力。其余测试点用来检查机器在动作中途卡住以后，模型能否继续给出有效命令。前五次判断只用来观察反应快慢，最终标准是二十次以内给出正确动作。

### 为什么测试使用记录中的真实状态

如果把模型动作不断累加来估算关节位置，即使回放人工驾驶动作，估算结果也会出现数弧度的误差。测试也无法生成与错误位置对应的四路相机画面。这样的图像、位置和速度并不来自同一个真实状态，用这种拼凑出来的状态评价模型，结果并不可信。

因此，离线测试使用实机记录中的真实图像、位置和速度。它能检查模型在这些真实状态下会给出什么命令，也能检查机器不动时模型会不会再次发力，但它不能代替真机运动轨迹测试。

### 模型要满足哪些条件

一套数据是开发期间使用的验证集，另一套是训练和调参都没有使用过的独立留出集。我们分别检查两套数据，标准如下：

1. 平均动作误差不得超过 0.13，动作方向一致率不得低于 90%。
2. 同一状态下，左区继续转、右区停下且两项都正确的比例不得低于 80%。目标在右区时，回转命令降到零的比例不得低于 95%。
3. 新模型在挖掘和回程阶段给出有效动作的比例，至少要达到人工驾驶记录对应比例的 80%。
4. 模型运行不能报错，动作数值必须正常，安全保护也不能被触发。

机器卡住测试使用同一批数据、同一批测试点和同一张死区表来比较新旧模型。新模型再次给出正确动作的比例不能低于原模型，给错方向或产生多余动作的次数也不能高于原模型。

测试代码包括：[完整记录和左右目标测试](../../testbed/testbed/cli/planner_open_loop_replay.py)、[机器卡住测试](../../testbed/testbed/cli/state_hold_transition_replay.py)和[结果汇总](../../testbed/testbed/cli/real_transition_acceptance.py)。

## 6. 离线测试得到什么结果

![模型听从左右目标，也保持了原有表现](svg/01-conclusion.svg)

### 模型已经会按照左右目标给出不同动作

开发验证集中，同一状态下左区继续转、右区停下且两项都正确的比例为 99.0%。独立留出集中的比例为 92.7%。两套数据中，目标在右区时，模型把回转命令降到零的比例都是 100%。这些结果说明，左右目标确实改变了模型直接给出的回转命令。

### 模型完成一铲的表现没有变差

![离线测试结果](svg/07-results.svg)

开发验证集中，新模型有 92.9% 的动作与人工驾驶动作方向一致。独立留出集中的比例为 91.6%。模型在挖掘和回程阶段的表现达到了预定标准，平均动作误差也在允许范围内。整个测试没有出现模型报错，也没有触发安全保护。

在机器保持不动的测试中，开发验证集里二十次以内再次给出正确动作的比例从原模型的 55.4% 提高到 67.5%。独立留出集中的比例从 58.4% 提高到 59.7%。两套数据中，模型第一次就给错方向或产生多余动作的测试点分别从 38 个降到 26 个、从 53 个降到 29 个。

完整记录还覆盖了左区到左区、左区到右区、右区到左区和右区到右区四种相邻目标组合。

### 现在可以得出什么结论

训练和离线测试证明，新增的左右回转损失能够让模型按照脚本给出的左右目标改变动作。模型完成一铲的整体表现没有变差。机器卡在死区时，它再次给出有效命令的表现也不低于原模型。

这些结果还不能说明液压系统一定会按预期动作，也不能证明机器会准确停进目标区或长时间稳定完成多铲。液压响应、实际落区和连续多铲运行还需要通过真机测试确认。
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
