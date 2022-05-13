import data_utils
import conv_model
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline

import tensorflow as tf
from keras.backend import learning_phase
from keras.utils.generic_utils import Progbar
import pdb
import pickle
input_size = 40
dataset_path = 'top_dataset.h5'
batch_size = 64
num_val_step = 10

model = conv_model.build()

input_data = tf.placeholder(tf.float32, (None, input_size, input_size, 2), name='input_data')
output_true = tf.placeholder(tf.float32, (None, input_size, input_size, 1), name='output_true')

with tf.variable_scope('topopt_model'):
    output_pred = model(input_data)

# metrics
correct_prediction = tf.equal(tf.round(output_true), tf.round(output_pred))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

experiments = {
    'uniform': {'model_path': '/content/notebooks/VariationalAutoencoder/uniform/VOL_COEFF=1.0'}, \
    'uniform_10': {'model_path': '/content/notebooks/VariationalAutoencoder/uniform/VOL_COEFF=10.0'}, \
    'uniform_50': {'model_path': '/content/notebooks/VariationalAutoencoder/uniform/VOL_COEFF=50.0'}, \
    'uniform_100': {'model_path': '/content/notebooks/VariationalAutoencoder/uniform/VOL_COEFF=100.0'}, \
    'poisson_5': {'model_path': '/content/notebooks/VariationalAutoencoder/poisson_5/VOL_COEFF=1.0'}, \
    }

iterations = range(5, 85, 5)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

for exp in experiments.values():
  saver.restore(sess, exp['model_path'])
  results = []
    
  progress = Progbar(max(iterations))
    
  for stop_iter in iterations:
    #print(stop_iter)
    progress.update(stop_iter)
    total_val_steps = 0
    current_results = []
        
    for x, y in data_utils.DatasetIterator(dataset_path, batch_size, lambda: stop_iter):
      if total_val_steps >= num_val_step:
        break
                
      feed_dict = {input_data: x, output_true: y, learning_phase(): 0}
      current_accuracy = sess.run(accuracy, feed_dict=feed_dict)
      current_results.append(current_accuracy)
      total_val_steps += 1
            
    results.append(np.mean(current_results))
        
  exp['results'] = results

results = []
progress = Progbar(max(iterations))

for stop_iter in iterations:
    progress.update(stop_iter)
    total_val_steps = 0
    current_results = []

    for x, y in data_utils.DatasetIterator(dataset_path, batch_size, lambda: stop_iter):
        if total_val_steps >= num_val_step:
            break

        y_pred = x[:, :, :, [0]] > 0.5
        current_accuracy = np.mean(y_pred == y)
        current_results.append(current_accuracy)
        total_val_steps += 1
        
    results.append(np.mean(current_results))
    
experiments['thresholding'] = {'results': results}

with open('saved_dictionary.pkl', 'wb') as f:
  pickle.dump(experiments, f)
print('DONE')