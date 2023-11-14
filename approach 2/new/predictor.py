import tensorflow as tf
import os
import pickle
import numpy as np

class MyPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor
        self._class_names = ["normal", "cancer"]
    
    def predict(self, instances, **kwargs):
        inputs = instances
        preprocessed_inputs = self._preprocessor.preprocess(inputs)
        outputs = self._model.predict(preprocessed_inputs)
        
        if kwargs.get('probablities'):
            return outputs.tolist()
        else:
            return [self._class_names[index] for index in np.argmax(outputs, axis=1)]
    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'ddsm.h5')
        model = tf.keras.models.load_model(model_path)
        
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        return cls(model, preprocessor)
        
        
