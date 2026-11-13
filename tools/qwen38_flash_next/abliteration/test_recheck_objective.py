"""CPU tests for recheck_objective.py on hand-made traces (standard library)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('recheck_objective', Path(__file__).with_name('recheck_objective.py'))
ro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ro)


def trace(*thoughts):
    return '\n\n'.join(thoughts)


def labels(reasoning, gold, final=None, **kw):
    return [(lab['j'], lab['label'], lab['reason']) for lab in ro.label_trace(reasoning, gold, final, **kw)[1]]


class LabelTest(unittest.TestCase):
    def test_split_keeps_exact_prefixes(self):
        text = 'A = 1.\n\nWait.\n\n\nB\n\n  \n\nC'
        spans = ro.split_thoughts(text)
        self.assertEqual(''.join(text[s:e] for s, e in spans), text)
        self.assertEqual([text[s:e] for s, e in spans], ['A = 1.\n\n', 'Wait.\n\n\n', 'B\n\n  \n\n', 'C'])

    def test_candidates(self):
        self.assertEqual(ro.thought_candidates('So the answer is $\\boxed{1,204}$.'), (1204, None, None))
        self.assertEqual(ro.thought_candidates('Total = 180 + 24 = 204 minutes'), (None, None, 204))
        self.assertEqual(ro.thought_candidates('x = 3 + y, so y = 7.'), (None, 7, 7))
        self.assertEqual(ro.thought_candidates('Thus m+n = 115 + 256 = 371.'), (None, 371, 371))
        self.assertEqual(ro.thought_candidates('a = 2.5 and b = 3x'), (None, None, None))
        self.assertEqual(ro.thought_candidates('Final answer: 12.'), (12, None, None))
        # An operator after the number: the sentence's result, or nothing.
        self.assertEqual(ro.thought_candidates('The answer is $657 + 64 = 721$.')[0], 721)
        self.assertEqual(ro.thought_candidates('final answer: $m + n = 115 + 256 = 371$')[0], 371)
        self.assertEqual(ro.thought_candidates('answer is $99/28$'), (None, None, None))
        self.assertEqual(ro.thought_candidates('the answer is \\(204\\).')[0], 204)
        self.assertEqual(ro.thought_candidates('The answer is 204 \\text{ minutes}.')[0], 204)
        self.assertEqual(ro.thought_candidates('The answer is 3x.')[0], None)
        self.assertEqual(ro.thought_candidates('so t = 24\\%.'), (None, 24, 24))
        self.assertEqual(ro.thought_candidates('x = 2 \\cdot 5'), (None, None, None))
        self.assertEqual(ro.thought_candidates('so n = 100\\,000.'), (None, 100000, 100000))
        self.assertEqual(ro.thought_candidates('k = 8 \\\\ next'), (None, None, 8))
        self.assertEqual(ro.thought_candidates('Final check: x is 100.'), (None, None, None))
        self.assertEqual(ro.thought_candidates('No numbers here.'), (None, None, None))

    def test_useful(self):
        r = trace('Compute: x = 4 + 5 = 9.', 'Wait, I dropped a term. So x = 12.', 'Then we finish.')
        self.assertEqual(labels(r, '12', 12), [(1, 'useful', None)])

    def test_harmful_needs_the_run_to_keep_the_wrong_answer(self):
        r = trace('So x = 12.', 'Hmm, maybe I double counted. So x = 10.', 'Then we finish.')
        self.assertEqual(labels(r, '12', 10), [(1, 'harmful', None)])
        self.assertEqual(labels(r, '12', 12), [(1, 'neutral', 'unconfirmed harmful')])
        self.assertEqual(labels(r, '12', None), [(1, 'neutral', 'unconfirmed harmful')])

    def test_verification_arithmetic_is_not_an_answer(self):
        r = trace('So x = 45.', 'Let me verify: 45 - 9 = 36 and 81 - 45 = 36.', 'Done.')
        self.assertEqual(labels(r, '45', 45), [(1, 'neutral', 'first confirmation')])

    def test_redundant_after_the_first_confirmation(self):
        r = trace('So x = 7.', 'Let me verify: fine.', 'Next.', 'Wait, check again: fine.',
                  'Hmm, one more look.', 'Done.')
        self.assertEqual(labels(r, '7', 7), [(1, 'neutral', 'first confirmation'),
                                              (3, 'redundant', None), (4, 'redundant', None)])
        self.assertEqual([x[1] for x in labels(r, '7', 7, redundant_after=2)],
                         ['neutral', 'neutral', 'neutral'])
        # Confirmations count per episode: the whole first check is kept.
        r = trace('So x = 7.', 'Let me verify: fine.', 'Alternatively, mod 7 gives the same.', 'Next.',
                  'Wait, again: fine.', 'Done.')
        self.assertEqual(labels(r, '7', 7), [(1, 'neutral', 'first confirmation'),
                                              (2, 'neutral', 'first confirmation'), (4, 'redundant', None)])

    def test_neutral_reasons(self):
        self.assertEqual(labels(trace('Start.', 'Wait, think.', 'Go on.'), '5'),
                         [(1, 'neutral', 'no candidate')])
        self.assertEqual(labels(trace('So y = 3.', 'Wait, so y = 4.', 'Go on.'), '5'),
                         [(1, 'neutral', 'wrong to wrong')])
        self.assertEqual(labels(trace('So y = 3.', 'Wait, recheck.', 'Go on.'), '5'),
                         [(1, 'neutral', 'no change')])
        # Back to a value already found: not a rescue.
        self.assertEqual(labels(trace('So y = 5.', 'Then z = 2.', 'Wait, so y = 5.', 'Go on.'), '5', 5),
                         [(2, 'neutral', 'returns to an earlier gold candidate')])

    def test_episode_runs_to_the_next_forward_thought(self):
        r = trace('So x = 12.', 'Wait, recheck.', 'Actually, so x = 10.', 'Continue.')
        spans, labs = ro.label_trace(r, '12', 10)
        self.assertEqual([(lab['j'], lab['e'], lab['label']) for lab in labs],
                         [(1, 3, 'harmful'), (2, 3, 'harmful')])


class BuildTest(unittest.TestCase):
    def records(self):
        problems = {'p1': {'id': 'p1', 'problem': 'P1?', 'level': 2024},
                    'p2': {'id': 'p2', 'problem': 'P2?', 'level': 2024}}
        r1 = trace('So x = 7.', 'Let me verify: fine.', 'Next step.', 'Wait, check again: fine.',
                   'Hmm, one more look.', 'Done, the answer is 7.')
        r2 = trace('So x = 12.', 'Hmm, maybe double counted. So x = 10.', 'So the answer is 10.')
        records = [dict(id='p1', answer='7', got='7', correct=True, finish_reason='stop', reasoning=r1,
                        content='The answer is $\\boxed{7}$.', usage={'completion_tokens': 50}),
                   dict(id='p2', answer='12', got='10', correct=False, finish_reason='stop', reasoning=r2,
                        content='$\\boxed{10}$', usage={'completion_tokens': 30})]
        return problems, records

    def test_rows(self):
        problems, records = self.records()
        pairs, keep, guard, report = ro.build([('s', problems, records, 'xhigh')], opener_tokens=3,
                                              keep_first=True)
        r1 = records[0]['reasoning']
        # p1: thought 3 and 4 are redundant (forward alternative: thought 5); p2: thought 1 harmful.
        self.assertEqual([(p['chain'], p['label']) for p in pairs],
                         [('s:p1', 'redundant'), ('s:p1', 'redundant'), ('s:p2', 'harmful')])
        first = pairs[0]
        self.assertEqual(first['propaganda'], 'Wait, check again: fine.\n\n')
        self.assertEqual(first['candid'], 'Done, the answer is 7.')
        self.assertEqual(first['reasoning'], r1[:r1.index('Wait')])
        self.assertTrue(first['reasoning'].endswith('Next step.\n\n'))
        self.assertEqual(first['prompt'], 'P1?' + ro.SUFFIX)
        self.assertEqual((first['score_reasoning'], first['reasoning_effort'], first['score_max_tokens']),
                         (True, 'xhigh', 3))
        self.assertEqual([(k['label'], k['answer']) for k in keep],
                         [('first-confirmation', 'Let me verify: fine.\n\n')])
        # The guard: only the right, finished run; its answer after the whole reasoning, closed.
        self.assertEqual(len(guard), 1)
        self.assertEqual((guard[0]['reasoning'], guard[0]['answer'], guard[0]['chain']),
                         (r1, records[0]['content'], 's:p1'))
        self.assertNotIn('score_reasoning', guard[0])
        counts = report['s']['counts']
        self.assertEqual((counts['redundant'], counts['harmful'], counts['useful'], counts['pairs'],
                          counts['keep'], counts['guard']), (2, 1, 0, 3, 1, 1))
        self.assertEqual(report['s']['cost']['chains'], 2)
        self.assertEqual(len({p['id'] for p in pairs + keep + guard}), 5)
        # Without the option, first confirmations are excluded.
        self.assertEqual(ro.build([('s', problems, records, 'xhigh')])[1], [])

    def test_boundary_cap_is_seeded(self):
        problems, records = self.records()
        many = trace('So x = 7.', *[f'Wait, check {i}: fine.\n\nStep {i}.' for i in range(12)], 'Done.')
        records[0]['reasoning'] = many
        a = ro.build([('s', problems, records[:1], 'medium')], max_boundaries=4)
        b = ro.build([('s', problems, records[:1], 'medium')], max_boundaries=4)
        self.assertEqual(len(a[0]), 4)
        self.assertEqual([p['id'] for p in a[0]], [p['id'] for p in b[0]])
        self.assertEqual(a[3]['s']['counts']['capped'], 11 - 4)

    def test_skips_non_integer_gold(self):
        problems, records = self.records()
        records[0]['answer'] = '\\frac{1}{2}'
        pairs, keep, guard, report = ro.build([('s', problems, records, 'low')])
        self.assertEqual(report['s']['counts']['non_integer_gold'], 1)
        self.assertEqual({p['chain'] for p in pairs}, {'s:p2'})
        with self.assertRaises(ValueError):
            ro.label_trace('a', '1/2')

    def test_refuses_aime2025_and_writes_files(self):
        problems, records = self.records()
        problems['p1']['level'] = 2025
        with self.assertRaises(SystemExit):
            ro.build([('s', problems, records, 'low')])
        problems['p1']['level'] = 2024
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'p.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in problems.values()))
            (d / 'r.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            self.assertEqual(ro.main(['--set', f's:{d}/p.jsonl:{d}/r.jsonl:medium', '--out', str(d / 'o')]), 0)
            pairs = [json.loads(l) for l in (d / 'o' / 'pairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(pairs), 3)
            self.assertIn('s', json.loads((d / 'o' / 'report.json').read_text()))
            with self.assertRaises(SystemExit):
                ro.main(['--set', f's:{d}/p.jsonl:{d}/r.jsonl:high', '--out', str(d / 'o')])


if __name__ == '__main__':
    unittest.main()
