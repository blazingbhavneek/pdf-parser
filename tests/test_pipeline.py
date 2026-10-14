"""Run: .venv/bin/python -m unittest discover -s tests -v"""

import math
import re
import unittest

from pdf_parser import charset, geometry, lattice, tables
from pdf_parser.pipeline import katakana_fixes
from pdf_parser.geometry import Block

# PaddleOCR-VL-1.6's real reply for a table with a two-row header (captured from vLLM).
OTSL = ("<fcel>フェーズ<fcel>材料<fcel>四分位統計<lcel><lcel><fcel>レンジ<lcel><nl><ucel><ucel><fcel>Q1"
        "<fcel>中央値<fcel>Q3<fcel>最小<fcel>最大<nl><fcel>概念設計<fcel>ポリカーボネート<fcel>70<fcel>78"
        "<fcel>85<fcel>62<fcel>90<nl><ucel><fcel>ステンレス<fcel>395<fcel>410<fcel>425<fcel>380<fcel>440<nl>")


def token(text, logprob, alternatives=()):
    data = text.encode() if isinstance(text, str) else text
    return {"token": "", "bytes": list(data), "logprob": logprob,
            "top_logprobs": [{"token": "", "bytes": list(a.encode()), "logprob": q} for a, q in alternatives]}


class PipelineTest(unittest.TestCase):
    def test_allowlist_keeps_japanese_and_blocks_simplified_forms(self):
        allowed = charset.japanese()
        self.assertTrue(set("統計値髙﨑①㈱ロットｶ") <= allowed)
        self.assertFalse(set("值说们这") & allowed)
        pattern = charset.regex_class(allowed) + "*"
        self.assertTrue(re.fullmatch(pattern, "統計値を確認]-^\\\n"))
        self.assertFalse(re.fullmatch(pattern, "統計值"))

    def test_logprob_bytes_rebuild_split_characters_and_same_length_alternatives(self):
        value = "値".encode()
        tokens = [token("統計", math.log(0.99), [("統許", math.log(0.01))]),
                  token(value[:2], math.log(0.9)), token(value[2:], math.log(0.6))]
        chars = lattice.from_logprobs("統計値", tokens, charset.japanese())
        self.assertEqual(lattice.text(chars), "統計値")
        self.assertEqual({c: round(p, 6) for c, p in chars[1].alts.items()}, {"計": 0.99, "許": 0.01})
        self.assertAlmostEqual(chars[2].p, 0.6)  # A split character keeps its weakest part.

    def test_merge_locks_agreement_and_slots_disagreement_and_gaps(self):
        vl = [lattice.Char(c, 0.9, {c: 0.9}) for c in "混合ロット"]
        ocr = [lattice.Char("混", 0.99), lattice.Char("合", 0.99), lattice.Char("口", 0.6, {"口": 0.6, "ロ": 0.3}),
               lattice.Char("ッ", 0.99), lattice.Char("卜", 0.7, {"卜": 0.7, "ト": 0.2}), lattice.Char("使", 0.99)]
        segs = lattice.segments(lattice.merge(vl, ocr), threshold=0.95)
        self.assertEqual(segs, ["混合", ("ロ", "口"), "ッ", ("ト", "卜"), ("", "使")])
        self.assertEqual(lattice.template(segs), "混合⟦ロ|口⟧ッ⟦ト|卜⟧⟦|使⟧")
        self.assertEqual(lattice.best(segs), "混合ロット")
        pattern = lattice.regex(segs)
        self.assertTrue(re.fullmatch(pattern, "混合ロット使"))
        self.assertTrue(re.fullmatch(pattern, "混合口ッ卜"))
        self.assertFalse(re.fullmatch(pattern, "混合ロツト"))  # Nothing outside the lattice.
        self.assertTrue(re.fullmatch(lattice.regex(["a.b(c)|[d]\n"]), "a.b(c)|[d]\n"))

    def test_otsl_spans_follow_paddlex_and_cells_keep_their_characters(self):
        chars = lattice.locked(OTSL)
        table = tables.parse_otsl(OTSL, chars)
        self.assertEqual((table.rows, table.cols), (4, 7))
        spans = {lattice.text(c.chars): (c.row, c.col, c.rowspan, c.colspan) for c in table.cells}
        self.assertEqual(spans["フェーズ"], (0, 0, 2, 1))
        self.assertEqual(spans["四分位統計"], (0, 2, 1, 3))
        self.assertEqual(spans["レンジ"], (0, 5, 1, 2))
        self.assertEqual(spans["概念設計"], (2, 0, 2, 1))
        html = tables.render(table, [tables.escape(lattice.text(c.chars)) for c in table.cells])
        rows = tables.parse_html(html)[0]
        self.assertEqual(rows[0][0], ("フェーズ", 2, 1, False))
        self.assertEqual(len(rows), 4)

    def test_table_constraint_lets_structure_change_but_never_text(self):
        table = tables.parse_otsl("<fcel>売上<fcel>髙<nl><fcel>1<fcel>2<nl>")
        segs = [lattice.segments(c.chars) for c in table.cells]
        segs[1] = [("髙", "高")]
        shown, pattern = tables.template_and_regex(table, segs)
        self.assertIn("⟦髙|高⟧", shown)
        best = tables.render(table, [tables.escape(lattice.best(s)) for s in segs])
        self.assertTrue(re.fullmatch(pattern, best))
        self.assertTrue(re.fullmatch(pattern, '<table><tr><td colspan="2">売上</td><td>高</td></tr>'
                                              '<tr><td>1</td><td>2</td><td></td></tr></table>'))
        self.assertFalse(re.fullmatch(pattern, "<table><tr><td>売上</td><td>高い</td></tr>"
                                               "<tr><td>1</td><td>2</td></tr></table>"))

    def test_grid_from_ocr_lines_keeps_a_column_the_table_model_dropped(self):
        def line(text, x, y):
            return {"box": (x, y, x + 20 * len(text), y + 20), "chars": lattice.locked(text)}

        lines = [line("品目0", 0, 0), line("0個", 200, 2), line("備考0", 400, 0),
                 line("品目1", 0, 40), line("10個", 200, 40), line("備考1", 400, 41)]
        dropped = tables.parse_otsl("<fcel>品目0<fcel>0個<nl><fcel>品目1<fcel>10個<nl>")
        ocr = [l["chars"] for l in lines]
        self.assertLess(tables.coverage(dropped, ocr), 0.9)
        grid = tables.from_lines(lines)
        self.assertEqual(tables.coverage(grid, ocr), 1.0)
        rows = tables.parse_html(tables.render(grid, [lattice.text(c.chars) for c in grid.cells]))[0]
        self.assertEqual([[c[0] for c in row] for row in rows], [["品目0", "0個", "備考0"], ["品目1", "10個", "備考1"]])

    def test_wrapped_cells_collect_their_lines_and_repeated_values_stay_in_their_rows(self):
        def line(text, x, y):
            return {"box": (x, y, x + 20 * len(text), y + 20), "chars": lattice.locked(text)}

        # The table reader misread both wrapped cells (阻帯, 解圧); the text layer holds the truth.
        table = tables.parse_otsl("<fcel>α-グルコシダーゼ阻帯薬<fcel>低<nl><fcel>腎臓での再吸収解圧<fcel>低<nl>")
        lines = [line("α-グルコシダー", 0, 0), line("低", 300, 10), line("ゼ阻害薬", 0, 25),
                 line("腎臓での", 0, 80), line("低", 300, 90), line("再吸収阻害", 0, 105)]
        tables.reconcile(table, lines, trusted=True)
        self.assertEqual([lattice.text(c.chars) for c in table.cells],
                         ["α-グルコシダーゼ阻害薬", "低", "腎臓での再吸収阻害", "低"])
        # 0円 must not take the 0 of 売上高0 and shift the column by a row.
        prices = tables.parse_otsl("<fcel>売上高0<fcel>0円<nl><fcel>売上高1<fcel>100円<nl>")
        tables.reconcile(prices, [line("売上高0", 0, 0), line("0 円", 300, 0), line("売上高1", 0, 40),
                                  line("100 円", 300, 40)], trusted=True)
        self.assertEqual([lattice.text(c.chars) for c in prices.cells], ["売上高0", "0 円", "売上高1", "100 円"])
        merged = tables.parse_otsl("<fcel>腎臓での再吸収解圧<nl>")
        tables.reconcile(merged, lines[3:])  # OCR lines only contest the table reader's characters.
        self.assertEqual(lattice.slots(lattice.segments(merged.cells[0].chars)), 2)

    def test_split_table_merges_and_drops_repeated_header(self):
        first = "<table><tr><th>項目</th><th>金額</th></tr><tr><td>売上</td><td>10</td></tr></table>"
        second = "<table><tr><td>項目</td><td>金額</td></tr><tr><td>利益</td><td>3</td></tr></table>"
        merged = tables.parse_html(tables.merge_split(first, second))[0]
        self.assertEqual([row[0][0] for row in merged], ["項目", "売上", "利益"])
        self.assertIsNone(tables.merge_split(first, "<table><tr><td>1</td></tr></table>"))

    def test_rare_voicing_variant_of_a_frequent_katakana_word_is_fixed(self):
        texts = ["ポリカーボネート"] * 3 + ["ボリカーボネートの検証", "ベース ペース ベース ベース ベース"]
        self.assertEqual(katakana_fixes(texts), {"ボリカーボネート": "ポリカーボネート"})

    def test_unordered_blocks_follow_the_block_above_them(self):
        title = Block("paragraph_title", (0, 0, 500, 30), 1)
        left = Block("text", (0, 40, 240, 300), 2)
        right = Block("text", (260, 40, 500, 300), 3)
        caption = Block("figure_title", (0, 320, 500, 340), None)
        table = Block("table", (0, 350, 500, 600), None)
        after = Block("text", (0, 620, 500, 700), 4)
        ordered = geometry.reading_order([table, after, caption, right, left, title])
        self.assertEqual(ordered, [title, left, right, caption, table, after])

    def test_skewed_lines_keep_their_order(self):
        angle = math.radians(6)

        def line(name, x, y, width=600, height=30):
            dx, dy = math.cos(angle), math.sin(angle)
            poly = [(x, y), (x + width * dx, y + width * dy),
                    (x + width * dx - height * dy, y + width * dy + height * dx), (x - height * dy, y + height * dx)]
            return {"name": name, "poly": poly, "box": geometry.bounds(poly)}

        lines = [line("second", 0, 45), line("first-b", 330, 35, 270), line("first-a", 0, 0, 300)]
        self.assertAlmostEqual(geometry.skew(lines), angle, places=3)
        ordered = geometry.order_lines(lines, lambda l: l["poly"], False, geometry.skew(lines))
        self.assertEqual([l["name"] for l in ordered], ["first-a", "first-b", "second"])


if __name__ == "__main__":
    unittest.main()
