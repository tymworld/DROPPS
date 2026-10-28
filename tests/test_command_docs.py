import unittest
from pathlib import Path

from scripts.update_command_docs import render_documents
from dropps.share.all_commands import all_commands
from dropps.share.command_docs import COMMAND_NOTES


class CommandDocumentationTests(unittest.TestCase):
    def test_every_public_command_has_current_complete_reference(self):
        root = Path(__file__).resolve().parents[1]
        documents = render_documents()
        self.assertEqual(set(COMMAND_NOTES), set(all_commands.names()))
        for name in all_commands.names():
            relative = Path("docs/commands") / f"{name}.md"
            with self.subTest(command=name):
                self.assertIn(relative, documents)
                self.assertIn("Examples:", documents[relative])
                self.assertIn(f"usage: dps {name}", documents[relative])
                self.assertEqual((root / relative).read_text(encoding="utf-8"), documents[relative])

    def test_published_distance_aliases_remain_available(self):
        for name in ("idist", "odist", "intra-distance", "inter-distance"):
            self.assertTrue(all_commands.exist(name))


if __name__ == "__main__":
    unittest.main()
