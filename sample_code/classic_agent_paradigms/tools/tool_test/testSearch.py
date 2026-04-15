import unittest
import sys
from pathlib import Path

# 将 tools 目录加入 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from search import search, quickSearch

class testSearch(unittest.TestCase):

    def testQuickSearch(self):
        print(quickSearch("what is 'Attention is all you need'?"))

    def testSearch(self):
        print(search("what is 'Attention is all you need'"))

if __name__ == '__main__':
    unittest.main()