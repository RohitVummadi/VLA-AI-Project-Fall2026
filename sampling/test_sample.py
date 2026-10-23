"""CPU-only contract checks; no model outputs or benchmark claims."""
import json
from pathlib import Path
import tempfile
import unittest

from sample import make_rows, parse_args, read_clip_ids


class SamplingContract(unittest.TestCase):
    def args(self, *extra):
        return parse_args(['--clip-ids', 'clips.txt', '--output', 'unused.jsonl', *extra])

    def test_clip_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clips.txt'
            path.write_text('# comment\na\n\nb # comment\n')
            self.assertEqual(read_clip_ids(path), ['a', 'b'])
            path.write_text('a\na\n')
            with self.assertRaises(ValueError):
                read_clip_ids(path)

    def test_schema_and_pairing(self):
        args = self.args('--K', '4')
        trajectories = [[[float(i), 0., 0.] for _ in range(64)] for i in range(4)]
        rows = list(make_rows(args, 'test-fixture', trajectories,
                             ['reason ' + str(i) for i in range(4)], {'runtime_s': 2.5}))
        self.assertEqual(len(rows), 4)
        for i, row in enumerate(rows):
            self.assertEqual(row['sample_id'], i)
            self.assertEqual(row['trajectory'][0][0], float(i))
            self.assertEqual(row['reasoning_text'], 'reason ' + str(i))
            self.assertEqual(row['runtime_s'], 2.5)
            self.assertIsNone(row['log_prob'])
            self.assertIsNone(row['perturbation'])
            json.dumps(row, allow_nan=False)

    def test_bad_shapes_fail(self):
        with self.assertRaises(ValueError):
            list(make_rows(self.args(), 'fixture', [[[0, 0, 0]]], ['text'], {'runtime_s': 1}))
        with self.assertRaises(ValueError):
            list(make_rows(self.args('--k', '4'), 'fixture', [], [], {'runtime_s': 1}))


if __name__ == '__main__':
    unittest.main()
