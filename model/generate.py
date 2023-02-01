import tensorflow as tf
from model.OneStep import OneStep 
from model.Model import Model
from model.Dataset import Dataset

def generate(input_string, length):
  dataset = Dataset()
  model = Model(
    vocab_size=dataset.get_vocab_length(),
    embedding_dim=256,
    rnn_units=512
    )
  model.load_weights(tf.train.latest_checkpoint('./model/training_checkpoints'))
  one_step_model = OneStep(
    model = model, 
    chars_from_ids = dataset.chars_from_ids, 
    ids_from_chars = dataset.ids_from_chars,
    temperature = 0.025
  )
  return one_step_model.generate(input_string, length)

if __name__ == '__main__':
  print(generate("Máj", 500))