import os
import tensorflow as tf

# Feature Engineering
def feature_engg_features(features):
  # 새로운 feature 추가
  features['distance'] = ((features['pickup_latitude'] - features['dropoff_latitude'])**2 +  (features['pickup_longitude'] - features['dropoff_longitude'])**2)**0.5
  features['trip_start_month'] = tf.strings.as_string(features['trip_start_month'])
  features['trip_start_hour'] = tf.strings.as_string(features['trip_start_hour'])
  features['trip_start_day'] = tf.strings.as_string(features['trip_start_day'])

  return(features)

def feature_engg(features, label):
  # feature 추가
  features = feature_engg_features(features)

  return(features, label)

def make_input_fn(dir_uri, mode, vnum_epochs = None, batch_size = 512):
    def decode_tfr(serialized_example):
      # 1. define a parser
      features = tf.io.parse_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
            'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
            'fare': tf.io.FixedLenFeature([], tf.float32),
            'pickup_latitude': tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
            'pickup_longitude': tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
            'trip_start_day': tf.io.FixedLenFeature([], tf.int64),
            'trip_start_hour': tf.io.FixedLenFeature([], tf.int64),
            'trip_start_month': tf.io.FixedLenFeature([], tf.int64)
        })

      return features, features['fare']

    def _input_fn(v_test=False):
      # Get the list of files in this directory (all compressed TFRecord files)
      tfrecord_filenames = tf.io.gfile.glob(dir_uri)

      # Create a `TFRecordDataset` to read these files
      dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

      if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = vnum_epochs # indefinitely
      else:
        num_epochs = 1 # end-of-input after this

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(buffer_size = batch_size)

      #Convert TFRecord data to dict
      dataset = dataset.map(decode_tfr)

      #Feature engineering
      dataset = dataset.map(feature_engg)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = vnum_epochs # indefinitely
          dataset = dataset.shuffle(buffer_size = batch_size)
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs)       
      
      #Begins - Uncomment for testing only -----------------------------------------------------<
      if v_test == True:
        print(next(dataset.__iter__()))
        
      #End - Uncomment for testing only -----------------------------------------------------<
      return dataset
    return _input_fn
