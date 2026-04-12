import unittest

from sample_code.simpleAgent.simpleAgent import *

class TestSimpleAgent(unittest.TestCase):
    def testGetWeather(self):
        weather = get_weather("Shanghai")
        print(weather)

    def testGetAttraction(self):
        weather = get_weather("Shanghai")
        attraction = get_attraction("Shanghai", weather)
        print(attraction)

    def testOpenAICompatibleClient(self):
        client = OpenAICompatibleClient()
        print(client.get_response('你好'))
        
if __name__ == '__main__':
    unittest.main()