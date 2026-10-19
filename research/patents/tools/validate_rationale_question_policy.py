"""Executable synthetic embodiment; confirmed reason extraction is an INPUT.

No language model, plant connection, or operating thresholds are implemented.
State labels and operation events are artificial. The selected-feature embodiment
reprojects past operation snapshots after learning an additional relevant item.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Observation:
    timestamp: int
    state: str
    usable: bool = True


@dataclass(frozen=True)
class Event:
    key: tuple
    timestamp: int
    observations: dict

    def state(self, tag, max_age=60):
        prior = [x for x in self.observations.get(tag, []) if x.timestamp <= self.timestamp]
        if not prior:
            return None
        latest = max(prior, key=lambda x: x.timestamp)
        if not latest.usable or self.timestamp - latest.timestamp > max_age:
            return None
        return latest.state


@dataclass(frozen=True)
class Answer:
    source_id: str
    event: Event
    relevant_tags: frozenset
    unresolved: bool = False


class Policy:
    def __init__(self):
        self.answers = []

    def learn(self, answer):
        # Only explicitly supplied answers can create knowledge; decide is read-only.
        if not answer.source_id or not answer.relevant_tags and not answer.unresolved:
            raise ValueError('An actual source and a relevant item or unresolved material are required')
        if any(x.source_id == answer.source_id for x in self.answers):
            raise ValueError('Duplicate answer source')
        self.answers.append(answer)

    def decide(self, event, draw=0.8):
        if not 0 <= draw < 1:
            raise ValueError('draw must be in [0, 1)')
        records = [x for x in self.answers if x.event.key == event.key]
        tags = sorted(set().union(*(x.relevant_tags for x in records)))
        reason, probability = '未取得の操作', 1.0
        if records:
            if any(x.unresolved for x in records):
                reason = '未対応の判断材料あり'
            else:
                current = tuple(event.state(tag) for tag in tags)
                if not tags or None in current:
                    reason = '現在の参照情報不足'
                else:
                    # No wildcard for a feature learned after the older record.
                    known = [tuple(x.event.state(tag) for tag in tags) for x in records]
                    complete = [signature for signature in known if None not in signature]
                    if current in complete:
                        reason, probability = '更新後の全参照項目で既知状態', 0.2
                    else:
                        reason = '更新後の全参照項目で未取得状態'
        return {'参照項目': tags, '判定': reason, '質問確率': probability,
                '質問出力': draw < probability, '今回理由': '未確認'}


KEY = ('系列X', 'PAC設定値', '増加')


def event(states, key=KEY, extra=None):
    values = {tag: [Observation(100, value)] for tag, value in states.items()}
    if extra:
        values.update(extra)
    return Event(key, 100, values)


def policy(source=None):
    p = Policy()
    p.learn(Answer('理由1', source or event({'濁度': '高', '水温': '低', 'pH': '通常'}),
                   frozenset({'濁度', '水温'})))
    return p


def main():
    rows = []

    def check(name, p, e, expected_ask, expected_tags, draw=0.8):
        count = len(p.answers)
        result = p.decide(e, draw)
        assert result['質問出力'] == expected_ask, (name, result)
        assert result['参照項目'] == sorted(expected_tags), (name, result)
        assert len(p.answers) == count and result['今回理由'] == '未確認'
        observed = {tag: [{'時刻': x.timestamp, '状態': x.state, '利用可能': x.usable}
                          for x in values] for tag, values in e.observations.items()}
        history = [{'回答出典': x.source_id, '参照項目': sorted(x.relevant_tags),
                    '未解決': x.unresolved,
                    '操作時点状態': {tag: x.event.state(tag) for tag in x.event.observations}}
                   for x in p.answers]
        rows.append({'ケース': name, '操作キー': json.dumps(e.key, ensure_ascii=False),
                     '操作時刻': e.timestamp, '監視入力': json.dumps(observed, ensure_ascii=False),
                     '理由履歴': json.dumps(history, ensure_ascii=False), '抽選値': draw,
                     '期待質問': expected_ask, **result, '照合': '一致'})

    same = event({'濁度': '高', '水温': '低', 'pH': '通常'})
    check('初回操作では質問', Policy(), same, True, [])
    check('二項目一致で質問低頻度化', policy(), same, False, ['濁度', '水温'])
    check('別設備の変化は除外', policy(), event({'濁度': '高', '水温': '低', '別設備': '異常'}), False, ['濁度', '水温'])
    check('関係する水温変化では質問', policy(), event({'濁度': '高', '水温': '通常'}), True, ['濁度', '水温'])
    check('参照項目欠測では質問', policy(), event({'濁度': '高'}), True, ['濁度', '水温'])
    check('最新値の品質不適合では質問', policy(), event({'濁度': '高'}, extra={'水温': [Observation(99, '低'), Observation(100, '低', False)]}), True, ['濁度', '水温'])
    check('古い値では質問', policy(), event({'濁度': '高'}, extra={'水温': [Observation(0, '低')]}), True, ['濁度', '水温'])
    check('操作後の値を混ぜない', policy(), event({'濁度': '高'}, extra={'水温': [Observation(100, '低'), Observation(110, '通常')]}), False, ['濁度', '水温'])
    check('設備が異なれば質問', policy(), event({'濁度': '高', '水温': '低'}, key=('系列Y', 'PAC設定値', '増加')), True, [])
    check('変更方向が異なれば質問', policy(), event({'濁度': '高', '水温': '低'}, key=('系列X', 'PAC設定値', '減少')), True, [])
    check('既知状態でも抽選で質問', policy(), same, True, ['濁度', '水温'], draw=0.1)

    p = policy()
    changed = event({'濁度': '高', '水温': '低', 'pH': '高'})
    check('追加前はpHを比較しない', p, changed, False, ['濁度', '水温'])
    p.learn(Answer('理由2', event({'濁度': '高', '水温': '低', 'pH': '低'}), frozenset({'pH'})))
    check('追加後のpH変化を検出', p, changed, True, ['pH', '濁度', '水温'])
    check('旧事例も追加項目で再比較', p, same, False, ['pH', '濁度', '水温'])
    check('新理由事例を再利用', p, event({'濁度': '高', '水温': '低', 'pH': '低'}), False, ['pH', '濁度', '水温'])

    p = policy(event({'濁度': '高', '水温': '低'}))
    p.learn(Answer('理由2', event({'濁度': '高', '水温': '低', 'pH': '低'}), frozenset({'pH'})))
    check('旧事例の追加項目欠測は任意一致にしない', p, same, True, ['pH', '濁度', '水温'])
    p = policy()
    p.learn(Answer('理由3', same, frozenset(), unresolved=True))
    check('未観測理由の取得後は省略を保留', p, same, True, ['濁度', '水温'])
    check('確率境界は厳密な未満', policy(), same, False, ['濁度', '水温'], draw=0.2)

    target = ROOT / 'sources' / '2026-10-14_rationale_question_policy_validation'
    target.with_suffix('.json').write_text(json.dumps({'scope': '合成状態・確認済み抽出結果を入力とする質問方針。自由文抽出は検証していない', 'cases': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    with target.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'cases': len(rows), 'result': 'passed', 'scope': 'synthetic confirmed-feature embodiment'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
