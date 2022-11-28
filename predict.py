import os
import numpy as np
import tensorflow as tf

class MyPredictor(object):
  def __init__(self, model, preprocessor):
    self._model = model
    self._preprocessor = preprocessor

  def predict(self, instances, **kwargs):
    inputs = np.asarray(instances)
    preprocessed_inputs = self._preprocessor.preprocess(inputs)
    outputs = self._model.predict(preprocessed_inputs)
    if kwargs.get('chunks'):
      return outputs.tolist()
    else:
      y_pred = outputs - 0.5
      y_pred *= 2
      weights = abs(y_pred)
      weights -= 0.2
      return (np.average(y_pred, weights=weights) / 2) + 0.5

  @classmethod
  def from_path(cls, model_dir):
    model_path = os.path.join(model_dir, 'ResNet.h5')
    model = tf.keras.models.load_model(model_path)



    return cls(model)