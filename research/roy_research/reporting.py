from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def write_utility_svg(path: Path, experiment: Dict[str, Any]) -> None:
    rows: List[Tuple[str, float]] = []
    for name, value in experiment.get("rule_arms", {}).items():
        rows.append((name, float(value["mean_utility"])))
    for name, value in experiment.get("learned_arms", {}).items():
        rows.append((name, float(value["test"]["mean_utility"])))
    width = 920
    row_height = 34
    top = 54
    bottom = 38
    label_width = 260
    chart_width = 610
    height = top + bottom + row_height * len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="24" y="30" font-family="system-ui,sans-serif" font-size="18" font-weight="700" fill="#202124">Controlled pilot mean task utility</text>',
    ]
    for index, (name, value) in enumerate(rows):
        y = top + index * row_height
        bar_width = max(0.0, min(1.0, value)) * chart_width
        color = "#315b7d" if name in experiment.get("learned_arms", {}) else "#9f6847"
        elements.extend([
            f'<text x="24" y="{y + 18}" font-family="ui-monospace,monospace" font-size="12" fill="#303438">{html.escape(name)}</text>',
            f'<rect x="{label_width}" y="{y + 4}" width="{bar_width:.2f}" height="20" rx="3" fill="{color}"/>',
            f'<text x="{min(width - 45, label_width + bar_width + 7):.2f}" y="{y + 19}" font-family="system-ui,sans-serif" font-size="12" fill="#202124">{value:.3f}</text>',
        ])
    elements.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
