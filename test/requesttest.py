import requests
def start():
    url="http://localhost:8002/frontEndControl/Start"
    data={"type":"KTZ66X32S-alltests-resolver",
          "serialNoList":["byd-alltests","byd-alltests-2"],
          "simu_count":2}
    r=requests.post(url,data=data)
    print(r)
def stop():
    url="http://localhost:8002/frontEndControl/Stop"
    r=requests.post(url)
    print(r)

if __name__=="__main__":
    # start()
    stop()