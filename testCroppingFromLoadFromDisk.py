from fashion_code.adapted_generators import SequenceFromDisk
from keras.applications.xception import Xception, preprocess_input

batch_size = 32
img_size = (299, 299)

train_gen = SequenceFromDisk('validation', batch_size, img_size,
                                     preprocessfunc=preprocess_input)

train_steps = len(train_gen)

for i in range(0, train_steps):
	print(train_gen[i])

