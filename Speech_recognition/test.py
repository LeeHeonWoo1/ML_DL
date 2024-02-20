import requests
import speech_recognition as sr

def sendCmd(cmd):       
    print(cmd)
    data = {'go': cmd}
    requests.get(action_uri, params=data)

esp_ip = "http://192.168.4.1"
action_uri = '{}:80/action'.format(esp_ip)
host = "{}:81/stream".format(esp_ip)

keyword_dict = {
    "forward" : "move to forward",
    "backward" : "move to backward",
    "left" : "move to left site",
    "right" : "move to right site"
    }

try:
    while True:
        r = sr.Recognizer()
        
        with sr.Microphone(device_index=1) as source:
            print("음성 입력 중입니다")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5, phrase_time_limit=3)                                          
            try:
                data = r.recognize_sphinx(audio)
                print(data)
            
                if "f" in data:
                    print('전진하겠습니다.')
                    sendCmd("forward")

                elif "b" in data:
                    print('후진하겠습니다.')
                    sendCmd("backward")

                elif "l" in data:
                    print('좌회전 하겠습니다.')
                    sendCmd("left")

                elif "r" in data:
                    print('우회전 하겠습니다.')
                    sendCmd("right")  
                    
                else:
                    print("멈추겠습니다.")
                    sendCmd("stop")
                
            except sr.UnknownValueError:
                print("이해하지 못했습니다.")
            except sr.RequestError:
                print("연결에 문제가 발생했습니다.")
                
except KeyboardInterrupt:
    pass
