from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from rq1.tasks.discovery import TaskDiscoveryError, discover_tasks
from rq1.tasks.selection import propose_manifest
from rq1.tasks.models import SelectionPolicy
from rq1.tasks.validation import overlap_errors, validate_manifest

NATIVE = [1, 6, 2, 3, 4, 5]

def write_task(root: Path, split: str, name: str, native: int) -> None:
    path=root/'json_2.1.1'/split/name; path.mkdir(parents=True)
    (path/'traj_data.json').write_text(json.dumps({'task_type': native}), encoding='utf-8')
    (path/'game.tw-pddl').write_text(json.dumps({'fixture': name}), encoding='utf-8')

class TaskManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        for index, native in enumerate(NATIVE): write_task(self.root, 'train', f't{index}', native); write_task(self.root, 'valid_seen', f's{index}', native)
        write_task(self.root, 'train', 't-extra', 1); write_task(self.root, 'valid_seen', 's-extra', 1)
    def tearDown(self): self.temp.cleanup()
    def test_discovery_is_deterministic_and_maps_all_families(self):
        first=discover_tasks(self.root, 'train'); second=discover_tasks(self.root, 'train')
        self.assertEqual(first.to_dict(), second.to_dict()); self.assertEqual(6, len({x.family for x in first.records}))
    def test_unseen_is_blocked_and_malformed_is_rejected(self):
        with self.assertRaises(TaskDiscoveryError): discover_tasks(self.root, 'valid_unseen')
        bad=self.root/'json_2.1.1'/'train'/'bad'; bad.mkdir(); (bad/'traj_data.json').write_text('bad'); (bad/'game.tw-pddl').write_text('{}')
        with self.assertRaises(TaskDiscoveryError): discover_tasks(self.root, 'train')
    def test_selection_is_seeded_balanced_and_manifest_is_not_frozen(self):
        discovery=discover_tasks(self.root, 'valid_seen'); policy=SelectionPolicy('v1', 7, 6)
        manifest=propose_manifest('pilot', discovery, policy, alfworld_version='0.4.2', repository_commit='commit')
        self.assertEqual([], validate_manifest(manifest)); self.assertEqual('proposed', manifest.status); self.assertEqual({1}, set(manifest.family_counts.values()))
        different=propose_manifest('pilot', discovery, SelectionPolicy('v1', 8, 6), alfworld_version='0.4.2', repository_commit='commit')
        self.assertNotEqual(manifest.manifest_sha256, different.manifest_sha256)
    def test_cross_manifest_overlap_is_rejected(self):
        discovery=discover_tasks(self.root, 'train'); policy=SelectionPolicy('v1', 1, 1)
        left=propose_manifest('acquisition', discovery, policy, alfworld_version=None, repository_commit=None)
        # Deliberately relabeling confirms identity checks do not rely on split names alone.
        right=propose_manifest('pilot', type(discovery)(discovery.schema_version, discovery.data_root_identity, 'valid_seen', discovery.records, discovery.exclusions, discovery.parse_errors), policy, alfworld_version=None, repository_commit=None)
        self.assertTrue(overlap_errors([left, right]))
