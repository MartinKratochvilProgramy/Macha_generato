import tensorflow as tf
import os
from generate import generate
from read_write_array import *
from Model import Model
from OneStep import OneStep
from Dataset import Dataset
from plot_history import plot_graphs

dataset = Dataset()
model = Model(
    vocab_size=dataset.get_vocab_length(),
    embedding_dim=256,
    rnn_units=512
    )

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
   optimizer='adam', 
   loss=loss, 
   )

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
try:
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    NEW_MODEL = False
    print("Weights loaded")
except:
    NEW_MODEL = True
    print("Weights not loaded")
   

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

STEPS = 1
for i in range(STEPS):
    print(f"STEP {i+1}/{STEPS}")
    history = model.fit(dataset.get_dataset(10_000, 64), epochs=1, callbacks=[checkpoint_callback])
    if NEW_MODEL:
        write_arr(history.history['loss'], './history/loss_history.csv')
        NEW_MODEL = False
    else:
        append_arr(history.history['loss'], './history/loss_history.csv')

    one_step_model = OneStep(
        model = model, 
        chars_from_ids = dataset.chars_from_ids, 
        ids_from_chars = dataset.ids_from_chars,
        temperature = 1
    )

    generated_text = one_step_model.generate("Viléme", 100)
    train_history = read_arr('./history/loss_history.csv')
    with open("output_history.txt", "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("\n")
        file.write(f"STEP: {len(train_history)}\n")
        file.write(generated_text)
    plot_graphs(train_history, 'loss')

    tf.saved_model.save(one_step_model, 'one_step')