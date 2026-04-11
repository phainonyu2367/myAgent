import unittest

from simpleAgent import *

class TestSimpleAgent(unittest.TestCase):
    def testGetWeather(self):
        weather = get_weather("Shanghai")
        print(weather)

    def testGetAttraction(self):
        weather = get_weather("Shanghai")
        attraction = get_attraction("Shanghai", weather)
        print(attraction)
        
if __name__ == '__main__':
    unittest.main()