import pickle
import requests
import os
import time
if __name__ == '__main__' :
    video_url = 'https://files.catbox.moe/tks0tu.mp4'
    labels = ['training', 'human', 'sport']
    api_url = 'http://194.163.165.205:7777/api/v1.1'
    print(api_url)
    req = requests.post(api_url + '/task', json={'video_url': video_url, 'labels': labels})
    if req.status_code != 200:
        print(req.text)
    id = req.json()['id']
    print(id)
    req = requests.get(api_url + f'/task/{id}/status')
    print(req.json())
    cnt = 0
    while True :
        req = requests.get(api_url + f'/task/{id}/status')
        if req.json()['status'] == 'ready' :
            data = requests.get(api_url + f'/task/{id}/data').json()
            print(data['data'])
            break
        else :
            cnt += 1
        time.sleep(1)
    print('it all takes {} seconds'.format(cnt))
