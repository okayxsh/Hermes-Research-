"""Parse actual documented command forms without invoking their handlers."""
import re
import shlex
import unittest
from pathlib import Path

from rq1.cli import build_parser


class MondayRunbookTests(unittest.TestCase):
    def test_all_documented_feasibility_commands_parse(self):
        text = (Path(__file__).resolve().parents[1] /
                "university-gpu/docs/NEXT_VISIT_RUNBOOK.md").read_text(encoding="utf-8")
        self.assertNotIn("python3.11 -m", text)
        self.assertIn('RQ1_PY="$PWD/.venv/bin/python"', text)
        parser = build_parser()
        commands = []
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for line in block.replace("\\\n", " ").splitlines():
                if "-m rq1.cli " not in line:
                    continue
                args = shlex.split(line.split("-m rq1.cli ", 1)[1])
                if "--cutoff-epoch" in args:
                    args[args.index("--cutoff-epoch") + 1] = "2000000000"
                with self.subTest(command=args):
                    parser.parse_args(args)
                commands.append(args[2])
        self.assertGreaterEqual(len(commands), 15)
        self.assertIn("export-progress", commands)
        self.assertIn("reference-library", commands)
