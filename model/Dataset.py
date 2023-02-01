import tensorflow as tf

class Dataset:
    def __init__(self) -> None:
        self.path_to_file = './model/maj.txt'
        self.text = (open(self.path_to_file, 'rb')
                .read()
                .decode(encoding='utf-8')
            )
        self.vocab = sorted(set(self.text))
        
        self.ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        
        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        
    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def get_vocab_length(self):
        return len(self.ids_from_chars.get_vocabulary())

    def get_dataset(self, BUFFER_SIZE, BATCH_SIZE):
        seq_length = 100

        all_ids = self.ids_from_chars(tf.strings.unicode_split(self.text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)

        dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
            )

        return dataset
    
    def get_ids(self, example_texts = []):
        chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
        ids = self.ids_from_chars(chars)
        return ids
    
    def get_chars(self, example_texts = []):
        chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary = self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        chars = chars_from_ids(self.get_ids(example_texts))

        return chars


if __name__ == '__main__':
    dataset = Dataset()
    print(dataset.get_ids(['abcdefg', 'xyz']))
    print(dataset.get_chars(['abcdefg', 'xyz']))