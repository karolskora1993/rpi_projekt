import requests

class RequestOrganizer:

    @staticmethod
    def sendRequest(url):
        files = {"image": open("image.png", "rb")}
        return requests.post(url, files=files)
        
