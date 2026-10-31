import ast,unittest
from prepare_assembly_copies import replace_one
class AdapterTests(unittest.TestCase):
    def test_anchor_once(self):self.assertEqual(replace_one('abc','b','d'),'adc')
    def test_missing_rejected(self):
        with self.assertRaises(ValueError):replace_one('abc','d','z')
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):replace_one('aaa','a','b')
if __name__=='__main__':unittest.main()
