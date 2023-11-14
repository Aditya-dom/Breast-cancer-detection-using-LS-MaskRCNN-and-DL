import googleapiclient.discovery
import cv2
import numpy as np
import base64


def explicit_compute_engine(project):
    from google.auth import compute_engine
    from google.cloud import storage

    # Explicitly use Compute Engine credentials. These credentials are
    # available on Compute Engine, App Engine Flexible, and Container Engine.
    credentials = compute_engine.Credentials()

    # Create the client using the credentials and specifying a project ID.
    storage_client = storage.Client(credentials=credentials, project=project)

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)


def predict_json(project, model, instances, version=None):
    # explicit()
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': [{'image_bytes': {'b64': instances}}]}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def load_image(image_str):
    img = cv2.imread(image_str)
    print(img)
    data = {}
    with open(image_str, mode='rb') as file:
        img = file.read()
    data['img'] = base64.encodebytes(img).decode("utf-8")
    instance = data['img']  # send this on server

    '''this is the logic to decode the image for preprocessing on model side
    img_str_to_bytes = bytes(instance, 'utf-8')
    with open('temp.jpg' , 'wb') as tmp: #converting bytes to temporary image on model side
        tmp.write(base64.decodebytes(img_str_to_bytes))
    
    image_to_numpy_preprocess = cv2.imread('temp.jpg')
    print(img_str_to_bytes)
    '''
    return instance


if __name__ == '__main__':
    model = 'CancerPredictor'
    instances = load_image('cancer.jpg')
    project = 'proud-storm-265122'
    version = 'v1'
    predict_json(project, model, instances, version)
    # explicit_compute_engine(project)
