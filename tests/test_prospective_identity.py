import copy
import json
import tempfile
import unittest
from pathlib import Path

from rq1.university_gpu.models import sha256_file
from rq1.university_gpu.prospective_identity import verify_candidate


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text("TEST TOKENIZER ONLY")
        self.link = self.root / "link.json"
        self.link.write_text("TEST LINK EVIDENCE ONLY")
        blob = self.root / "blob"
        blob.write_bytes(b"test model layer")
        digest = sha256_file(blob)
        self.blob = self.root / ("sha256-" + digest)
        blob.rename(self.blob)
        self.manifest = self.root / "manifest.json"
        ref = {"digest": "sha256:" + digest, "size": self.blob.stat().st_size}
        self.manifest.write_text(json.dumps({"config": ref, "layers": [ref]}))
        self.provenance = {"model_manifest_sha256": sha256_file(self.manifest),
            "tokenizer_sha256": sha256_file(self.tokenizer), "revision": "a" * 40,
            "source_repository": "https://example.test/test-fixture", "model_tokenizer_link_evidence_sha256": sha256_file(self.link),
            "samples": [{"text": text, "token_ids": list(text.encode()), "reference_source": "fake-test-reference"}
                        for text in ["", "hello", "é", "{}", "task"]]}

    def verify(self, provenance=None):
        return verify_candidate(self.manifest, self.root, self.tokenizer, provenance or self.provenance,
                                lambda text: list(text.encode()), self.link)

    def test_valid_evidence_never_authorizes_launch(self):
        result = self.verify()
        self.assertEqual("CANDIDATE_REVIEW_REQUIRED", result["status"])
        self.assertFalse(result["approved"])
        self.assertFalse(result["launch_authorized"])

    def test_blob_corruption_blocks(self):
        self.blob.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "blob missing or corrupt"):
            self.verify()

    def test_manifest_replacement_blocks(self):
        self.manifest.write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.verify()

    def test_reference_mismatch_blocks(self):
        p = copy.deepcopy(self.provenance)
        p["samples"][0]["token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "reference token comparison"):
            self.verify(p)

    def test_link_evidence_tampering_blocks(self):
        self.link.write_text("changed")
        with self.assertRaisesRegex(ValueError, "link evidence"):
            self.verify()

    def test_tokenizer_replacement_blocks(self):
        self.tokenizer.write_text("different")
        with self.assertRaisesRegex(ValueError, "tokenizer source"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
