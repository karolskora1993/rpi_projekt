import urllib
import urllib2

class RequestOrganizer:

    @staticmethod
    def sendRequest(url, send_data):
        data = urllib.urlencode(send_data)
        request = urllib2.Request(url, data)
        response = urllib2.urlopen(request).read()

        return response
