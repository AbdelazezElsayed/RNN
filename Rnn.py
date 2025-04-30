import numpy as np

words = ['I', 'like', 'to', 'code']
word_to_idx = {w: i for i, w in enumerate(words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

vocab_size = len(words)
hidden_size = 10
lr = 0.1
epochs = 200

def one_hot(idx):
    v = np.zeros((vocab_size, 1))
    v[idx] = 1
    return v

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

x_indices = [word_to_idx['I'], word_to_idx['like'], word_to_idx['to']]
target = word_to_idx['code']

for e in range(epochs):
    hs = [np.zeros((hidden_size, 1))]
    xs = []

    for i in x_indices:
        x = one_hot(i)
        xs.append(x)
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs[-1]) + bh)
        hs.append(h)

    y = np.dot(Why, hs[-1]) + by
    probs = np.exp(y) / np.sum(np.exp(y))
    loss = -np.log(probs[target])

    dy = probs
    dy[target] -= 1

    dWhy = np.dot(dy, hs[-1].T)
    dby = dy
    dh = np.dot(Why.T, dy)

    for t in reversed(range(3)):
        dtanh = (1 - hs[t+1] ** 2) * dh
        dWxh = np.dot(dtanh, xs[t].T)
        dWhh = np.dot(dtanh, hs[t].T)
        dbh = dtanh

        Wxh -= lr * dWxh
        Whh -= lr * dWhh
        bh -= lr * dbh

        dh = np.dot(Whh.T, dtanh)

    Why -= lr * dWhy
    by -= lr * dby

h = np.zeros((hidden_size, 1))
for i in x_indices:
    x = one_hot(i)
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

y = np.dot(Why, h) + by
pred = np.argmax(y)

print("Predicted word:", idx_to_word[pred])
print("Abdelazez#####")
