import keras
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences


chr_to_int = {str(i): i for i in range(10)}
ops = ['+', '*', '<', '>']
for op in ops:
    chr_to_int[op] = len(chr_to_int)

val_set = {'6*4', '8*3', '9+2', '5+7', '9<8', '9*9'}

EMBEDDING_SIZE = 10
N_UNITS = 10
MAX_OUT = 10000
N_SAMPLES = 100000
N_DIGITS = 1
MAX_IN = 10**N_DIGITS - 1

inp = keras.layers.Input([None])

emb = keras.layers.Embedding(len(chr_to_int), EMBEDDING_SIZE)(inp)
out = keras.layers.LSTM(N_UNITS, return_sequences=False)(emb)
p = keras.layers.Dense(MAX_OUT, activation='softmax')(out)

outputs = [p]
inputs = [inp]

model = keras.models.Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['acc'])

def data():
    xs = []
    ys = []
    while len(xs) < N_SAMPLES:
        x1 = random.randint(0, MAX_IN)
        x2 = random.randint(0, MAX_IN)
        op = random.sample(ops, 1)[0]
        # fill with zeros, 5 -> 005
        x1 = str(x1).rjust(N_DIGITS, '0')
        x2 = str(x2).rjust(N_DIGITS, '0')

        s = x1 + op + x2
        # Skip val_set:
        if s in val_set:
            continue

        y = eval(s)
        
        x = [chr_to_int[c] for c in s]

        xs.append(x)
        ys.append(y)

    xs = pad_sequences(xs)
    ys = np.array(ys)

    return xs, ys

xs, ys = data()
xs_val, ys_val = data()

model.fit(xs, ys, epochs=100)
print(model.evaluate(xs_val, ys_val))

while True:
    i = input('Give me an expression\n')
    if len(i) == 0:
        break
    x = [chr_to_int[c] for c in i]
    yhat = model.predict(np.array([x]))
    print('Prediction:', yhat.argmax())
