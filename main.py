import traceback
from flask import Flask
import flask
from flask import Response
import random
import numpy as np
import time
import os
import yaml
import threading
import cv2
import logging

import torch
import clip
from PIL import Image

id_results = {}
id_busy = {}


def get_proba(image_batch, labels):
    with torch.no_grad():
        pil_image_batch = []
        for image_cv2 in image_batch:
            image = preprocess(Image.fromarray(image_cv2)).unsqueeze(0).to(device)
            pil_image_batch.append(image)

        text = clip.tokenize(labels).to(device)

        # image_features = model.encode_image(pil_image_batch)
        # text_features = model.encode_text(text)
        pil_image_batch = torch.cat(pil_image_batch)
        logging.info(pil_image_batch.shape)
        logits_per_image, logits_per_text = model(pil_image_batch, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs


def get_mediapipe_landmark(video_path, labels):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    freq = cap.get(cv2.CAP_PROP_FPS) // CONFIG['fps']
    frame_ind = 0

    sum_proba = np.zeros(len(labels))
    batch = []
    while cap.isOpened():
        success, image = cap.read()
        if (len(batch) % CONFIG['batch_size'] == 0 or not success) and len(batch) != 0:
            batch_proba = get_proba(batch, labels)
            for proba in batch_proba:
                sum_proba += proba
            batch = []

        if not success:
            break

        frame_ind += 1
        if frame_ind % freq != 0:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch.append(image)

    cap.release()
    end_time = time.time()
    logging.info('done ' + video_path + ' in {} seconds'.format(round(end_time - start_time, 2)))

    tmp = [(sum_proba[i], labels[i]) for i in range(len(labels))]
    tmp.sort(key=lambda x: -x[0])
    result = [{'text': el[1], 'sum_proba': el[0]} for el in tmp[:5]]
    return result


app = Flask(__name__)


def mythread(data, id):
    id_busy[id] = 1

    logging.info('downloading' + data['video_url'])
    video_format = data['video_url'][-3:]
    video_name = 'video{}.'.format(random.randint(0, 15)) + video_format
    if os.path.exists(video_name):
        os.remove(video_name)
    os.system('wget -O {} {}'.format(video_name, data['video_url']))

    logging.info('detecting ' + str(video_name))
    top_labels = get_mediapipe_landmark(video_path=video_name, labels=data['labels'])
    id_results[id] = top_labels

    id_busy[id] = 0


@app.route('/task', methods=["GET", "POST"])
def predict_video():
    data = flask.request.json
    id = random.randint(0, 1e18)
    thr = threading.Thread(target=mythread, args=(data, id, ))
    thr.start()

    return {'id': id}


@app.route('/task/<id>/status', methods=["GET"])
def is_busy(id):
    id = int(id)
    if not id in id_busy.keys():
        return Response("Task with this id doesn't exist", status=400)

    if id_busy[id] == 0:
        return {'status': 'ready'}
    else:
        return {'status': 'in_queue'}


@app.route('/task/<id>/data', methods=["GET"])
def get_data(id):
    id = int(id)
    if not id in id_busy.keys():
        return Response("Task with this id doesn't exist", status=400)

    if id_busy[id] == 1:
        return Response("Task with this id isn't ready", status=425)

    return {'data': id_results[id]}


if __name__ == '__main__':
    mydir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(mydir, 'config.yaml'), "r") as config_file:
        CONFIG = yaml.safe_load(config_file)

    logging.basicConfig(filename=os.path.join(mydir, 'sample.log'),
                        level='INFO',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    app.run(host='0.0.0.0', port=CONFIG['port'])

