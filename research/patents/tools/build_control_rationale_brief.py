"""Render the research brief from its Markdown source and record synthetic checks."""
from pathlib import Path
import csv
import html
import json
import re
from html.parser import HTMLParser
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
STEM = 'BRAINSTORM_CONTROL_RATIONALE'

CASES = [
    ('条件成立', [(-10, 12, True), (0, 15, True)], '通常の二条件', '成立'),
    ('無関係項目の変化', [(-10, 12, True), (0, 15, True)], '水温だけが変化。条件は濁度のみ', '成立'),
    ('同じ現在値・異なる傾向', [(-10, 15, True), (0, 15, True)], '現在値は同じだが上昇していない', '不成立'),
    ('比較期間の欠測', [(0, 15, True)], '10分前の値がない', '評価不能'),
    ('操作時点の品質不足', [(-10, 12, True), (0, 15, False)], '操作時点の値を利用できない', '評価不能'),
    ('操作後の値が変化', [(-10, 12, True), (0, 15, True), (5, 8, True)], '5分後の値を判定に混ぜない', '成立'),
    ('比較基準と同値', [(-10, 7, True), (0, 10, True)], '「超える」なので同値は成立しない', '不成立'),
    ('既知の不成立と欠測', [(0, 8, True)], '高さが不成立。傾向不明でもAND全体は不成立', '不成立'),
]


def evaluate(data):
    # Deliberately require the defined observation instants; no interpolation.
    samples = data['原水濁度']
    values = {t: v for t, v, valid in samples if valid and t <= 0}
    current, earlier = values.get(0), values.get(-10)
    high = None if current is None else current > 10
    rising = None if current is None or earlier is None else current - earlier > 2
    if high is False or rising is False:
        return '不成立'
    if high is True and rising is True:
        return '成立'
    return '評価不能'


def validate():
    rows = []
    for name, samples, purpose, expected in CASES:
        data = {'原水濁度': samples, '水温': 35 if name == '無関係項目の変化' else 20}
        actual = evaluate(data)
        if actual != expected:
            raise ValueError(f'{name}: expected {expected}, got {actual}')
        rows.append([name, json.dumps(data, ensure_ascii=False), purpose, expected, actual, '一致'])
    output = ROOT / 'sources' / '2026-10-13_control_rationale_validation.csv'
    output.parent.mkdir(exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ケース', '監視入力（系列は操作相対分,値,品質適合）', '確認点', '期待評価', '計算評価', '照合結果'])
        writer.writerows(rows)
    table = ['| 仮想ケース | 確認点 | 評価結果 |', '|---|---|---|']
    table += [f'| {r[0]} | {r[2]} | {r[4]} |' for r in rows]
    table.append('\n8例で期待評価と計算評価が一致した。これは合成データの機能確認である。')
    return '\n'.join(table)


def svg_base(name, height, content, description):
    return f'''<figure><svg viewBox="0 0 900 {height}" role="img" aria-labelledby="{name}-title {name}-desc">
<title id="{name}-title">{html.escape(name)}</title><desc id="{name}-desc">{html.escape(description)}</desc>
<defs><marker id="{name}-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#222"/></marker></defs>
<g class="diagram" style="--arrow:url(#{name}-arrow)">{content}</g></svg><figcaption>{html.escape(name)}</figcaption></figure>'''


ACQUISITION = svg_base('図1　回答から次回の判断条件を得る', 125, '''
<rect x="15" y="28" width="240" height="66"/><text x="135" y="54"><tspan>操作した理由を</tspan><tspan x="135" dy="23">運転員に聞く</tspan></text>
<path class="line" d="M255 61H330"/>
<rect x="330" y="28" width="240" height="66"/><text x="450" y="54"><tspan>回答から項目と</tspan><tspan x="450" dy="23">条件を取り出す</tspan></text>
<path class="line" d="M570 61H645"/>
<rect x="645" y="28" width="240" height="66"/><text x="765" y="54"><tspan>暗黙知DBに</tspan><tspan x="765" dy="23">理由と条件を追加</tspan></text>
''', '操作理由を聞き、回答から判断項目と条件を取り出し、暗黙知DBへ操作と関連付けて保存する。')

RUNTIME = svg_base('図2　判断条件を次回の質問に反映する', 520, '''
<rect x="300" y="10" width="300" height="56"/><text x="450" y="44">同じ種類の操作を検出</text>
<path class="line" d="M450 66V94"/>
<rect x="300" y="94" width="300" height="56"/><text x="450" y="117"><tspan>必要な測定値を</tspan><tspan x="450" dy="22">操作時点で確認</tspan></text>
<path class="line" d="M450 150V180"/>
<polygon points="450,180 595,245 450,310 305,245"/><text x="450" y="238"><tspan>DBの判断条件に</tspan><tspan x="450" dy="23">当てはまるか</tspan></text>
<path class="line" d="M595 245H730V345"/><text class="label" x="670" y="230">はい</text>
<rect x="615" y="345" width="230" height="60"/><text x="730" y="380">質問する回数を減らす</text>
<path class="line" d="M305 245H170V345"/><text class="label" x="185" y="230">いいえ・値が不明</text>
<rect x="55" y="345" width="230" height="60"/><text x="170" y="380">今回の理由を聞く</text>
<path class="line" d="M170 405V465H300"/>
<path class="line" d="M730 405V465H600"/><text class="label" x="730" y="440">回答を得た場合</text>
<rect x="300" y="435" width="300" height="60"/><text x="450" y="459"><tspan>回答を使い、暗黙知DBの</tspan><tspan x="450" dy="22">理由と条件を補う</tspan></text>
''', '保存した条件に当てはまれば質問を減らし、それ以外や測定値が不明な場合は理由を聞く。新たな回答から項目と条件を更新する。')

CSS = '''
:root{color-scheme:light;--ink:#202020;--line:#bababa;--muted:#666}
*{box-sizing:border-box}body{margin:0;background:#f3f3f1;color:var(--ink);font-family:"Yu Mincho","Hiragino Mincho ProN",serif;font-size:15px;line-height:1.85}
nav{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #ccc;padding:10px 24px;display:flex;align-items:center;justify-content:center;gap:16px;font-family:"Yu Gothic",sans-serif;font-size:13px}
button,select{font:inherit;color:inherit;background:#fff;border:1px solid #aaa;border-radius:2px;padding:6px 10px}button:disabled{opacity:.35}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid #444;outline-offset:3px}select{max-width:45vw}
main{max-width:1080px;margin:24px auto 48px}.page{background:#fff;padding:36px 58px 28px;min-height:850px;border:1px solid #ddd}.page[hidden]{display:none}.page-footer{margin-top:28px;padding-top:10px;border-top:1px solid #ccc;color:var(--muted);font:12px "Yu Gothic",sans-serif;text-align:right}
h1{font-size:28px;line-height:1.5;font-weight:600;margin:0 0 8px}h2{font-size:20px;margin:22px 0 14px;border-bottom:1px solid #444;padding-bottom:5px;line-height:1.6}h2:first-child{margin-top:0}h3{font-size:16px;margin:22px 0 8px}p{margin:0 0 13px}h1+p{font-size:12px;color:var(--muted)}strong{font-weight:600}a{color:#214b70;text-underline-offset:3px}table{width:100%;border-collapse:collapse;margin:16px 0;font:13px/1.7 "Yu Gothic",sans-serif}th,td{border:1px solid var(--line);padding:9px 10px;vertical-align:top;text-align:left;overflow-wrap:anywhere}th{background:#f3f3f3;font-weight:600}th:first-child{min-width:145px}figure{margin:22px 0}svg{width:100%;height:auto;display:block}figcaption{font:12px "Yu Gothic",sans-serif;text-align:center;color:var(--muted);margin-top:6px}.diagram rect,.diagram polygon{fill:#fff;stroke:#222;stroke-width:1.4}.diagram text{font:16px "Yu Gothic",sans-serif;fill:#222;text-anchor:middle}.diagram .label{font-size:13px}.diagram .line{fill:none;stroke:#222;stroke-width:1.5;marker-end:var(--arrow)}
@media(max-width:750px){nav{gap:8px;padding:8px}main{margin:0 auto}.page{padding:24px 20px;min-height:0;border:0}h1{font-size:24px}table{font-size:12px}th,td{padding:7px}th:first-child{min-width:90px}.page-footer{margin-top:20px}}
@media print{@page{size:A4;margin:14mm}body{background:#fff;font-size:10pt;line-height:1.7}nav{display:none}main{max-width:none;margin:0}.page,.page[hidden]{display:block;border:0;padding:0;min-height:0;break-after:page}.page:last-child{break-after:auto}h1{font-size:20pt}h2{font-size:14pt;margin:14px 0 9px}h3{font-size:11pt}p{margin-bottom:9px}table{font-size:8.5pt;line-height:1.55;margin:10px 0}th,td{padding:6px}figure{break-inside:avoid;margin:12px 0}thead{display:table-header-group}tr{break-inside:avoid}a{color:inherit;text-decoration:none}.page-footer{margin-top:14px;font-size:8pt}}
'''

CSS += '''
main{background:#fff;border:1px solid #ddd}
.page{border:0;min-height:0;padding-top:28px;padding-bottom:12px}
.page + .page{border-top:1px solid #ddd}
.page:last-child{padding-bottom:36px}
@media(max-width:750px){main{border:0}}
'''


class StructureCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.counts = {'article': 0, 'svg': 0, 'table': 0}
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag in self.counts:
            self.counts[tag] += 1
        if tag == 'a':
            self.links.append(attrs.get('href', ''))


def main():
    source = (ROOT / f'{STEM}.md').read_text(encoding='utf-8')
    renderer = MarkdownIt('commonmark', {'html': True}).enable('table')
    sections = source.split('<!-- pagebreak -->')
    assert len(sections) == 5
    pages = []
    for i, section in enumerate(sections):
        for name, figure in [('acquisition',ACQUISITION),('runtime',RUNTIME)]:
            section = re.sub(r'<!-- figure:'+name+r' -->\s*```mermaid[\s\S]*?```', lambda _:figure,section)
        body = renderer.render(section)
        pages.append(f'<article class="page" id="page-{i+1}">{body}</article>')
    document = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(source.splitlines()[0].lstrip('# '))}</title><style>{CSS}</style></head><body>
<main>{''.join(pages)}</main></body></html>'''
    (ROOT / f'{STEM}.html').write_text(document, encoding='utf-8')
    check = StructureCheck()
    check.feed(document)
    assert check.counts == {'article': 5, 'svg': 2, 'table': 4}, check.counts
    assert len(check.ids) == len(set(check.ids)), 'Duplicate HTML IDs'
    assert len(check.links) >= 11 and all(link.startswith('https://') for link in check.links)
    assert '```' not in document and '<!-- figure:' not in document
    assert '<button' not in document and '<script' not in document and ' hidden' not in document
    print(json.dumps({'pages':len(pages),'structure':'passed','html':str(ROOT/f'{STEM}.html')},ensure_ascii=True))


if __name__ == '__main__':
    main()
