import unittest

from simpleAgent import *

class TestSimpleAgent(unittest.TestCase):
    def testGetWeather(self):
        weather = get_weather("Shanghai")
        print(weather)
        
if __name__ == '__main__':
    unittest.main()