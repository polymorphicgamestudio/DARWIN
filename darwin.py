"""
https://karpathy.ai/zero-to-hero.html
NOTE(caleb): Probabilistic language model impl. from ^^ nn zero to hero ^^
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""
import random
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm # Progress bars

def splash(): 
    sys.stdout.write("\033c") # NOTE(caleb): OS agnostic way to clear screen
    print("""
     ______   ________   ______    __ __ __    ________  ___   __      
    /_____/\ /_______/\ /_____/\  /_//_//_/\  /_______/\/__/\ /__/\    
    \:::_ \ \\\\::: _  \ \\\\:::_ \ \ \:\\\\:\\\\:\ \ \__.::._\/\::\_\\\\  \ \   
     \:\ \ \ \\\\::(_)  \ \\\\:(_) ) )_\:\\\\:\\\\:\ \   \::\ \  \:. `-\  \ \  
      \:\ \ \ \\\\:: __  \ \\\\: __ `\ \\\\:\\\\:\\\\:\ \  _\::\ \__\:. _    \ \ 
       \:\/.:| |\:.\ \  \ \\\\ \ `\ \ \\\\:\\\\:\\\\:\ \/__\::\__/\\\\. \`-\  \ \\
        \____/_/ \__\/\__\/ \_\/ \_\/ \_______\/\________\/ \__\/ \__\/ v0.0.2
                        (now with probabilistic modeling)
    """)

"""
Given mappings of tokens <=> indicies, and a dataset (list of mcdonald's reviews for now) and a
target block size. Return 2 tensors X and Y where X is 'n_inputs x block_size' representing all contexts 
for the input data set and Y is the 'label' or expected output for that context. 
Example: block_size = 3 and n_inputs >= 1 -- X[0]: (7, 1, 12) => Y[0]: 5
"""   
def build_dataset(stoi: dict, itos: dict, data: [str], block_size: int):
    X, Y = [], []
    for chars in data:
        context = [0] * block_size
        for ch in chars + itos[0]:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '=>', itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

splash()

mc_path = 'data/clean_McDonald_s_Reviews.txt'
special = '🔥' # NOTE(caleb): Some unique identifier (that isn't in vocab)
dirty_reviews = open(mc_path, 'r').read().splitlines()
mc_reviews = [] 
for review in dirty_reviews:
    clean_review = ''
    for ch in review:
        if ord(ch) < 128:
            clean_review += ch
    mc_reviews.append(clean_review)

print('Read:', mc_path)
vocab = sorted(list(set(''.join(mc_reviews)))) # Unique characters in data set
vocab_size = len(vocab)
print('Initialized vocab:',''.join(vocab))

stoi = { s:i+1 for i,s in enumerate(vocab) } # Mapping from characters to integers
assert(special not in stoi)  
stoi[special] = 0 # NOTE(caleb): THIS MUST BE ZERO
vocab_size += 1
itos = { i:s for s,i in stoi.items() } # Map back to characters
encode = lambda s: [stoi[c] for c in s] # Take a string, output list of characters 
decode = lambda l: ''.join([itos[i] for i in l]) # Take a list of integers, output a string

feature_length = 30 # Features per context  
block_size = 7 # Context length: how many characters do we take to predict next one?
hlayer_nrons = 200

random.shuffle(mc_reviews)
n1 = int(0.8*len(mc_reviews))
n2 = int(0.9*len(mc_reviews))
Xtr, Ytr = build_dataset(stoi, itos, mc_reviews[:n1], block_size)
Xdev, Ydev = build_dataset(stoi, itos, mc_reviews[n1:n2], block_size)
Xte, Yte = build_dataset(stoi, itos, mc_reviews[n2:], block_size)

# Make this script deterministic
g = torch.Generator().manual_seed(2147483647) 

C = torch.randn((vocab_size, feature_length), generator=g, requires_grad=True) # Context matrix mapping vocab indicies => feature vec
W1 = torch.randn((block_size * feature_length, hlayer_nrons), generator=g, requires_grad=True)
b1 = torch.randn(hlayer_nrons, generator=g, requires_grad=True)
W2 = torch.randn((hlayer_nrons, vocab_size), generator=g, requires_grad=True)
b2 = torch.randn(vocab_size, generator=g, requires_grad=True)
params = [C, W1, b1, W2, b2]

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
# lri = []
# lossi = []
# stepi = []

# Gradient descent
for i in tqdm(range(200000)):
    # Minibatch construct)
    ix = torch.randint(0, Xtr.shape[0], (vocab_size,))
    
    # Forward pass
    emb = C[Xtr[ix]]
    print(emb)
    break 
    # emb = torch.zeros((len(X), block_size, feature_length))
    # for x_index, context in enumerate(X):
    #     for context_index, tok_index in enumerate(context):
    #         emb[x_index][context_index] = C[tok_index]
    h = torch.tanh(emb.view(-1, block_size * feature_length) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    # print(loss.item())
    # NOTE(caleb): cross_entropy is doing next 3 lines under the hood. 
    # counts = logits.exp() # Something that looks like counts (N matrix)
    # prob = counts / counts.sum(1, keepdims=True) # Probabilities for next character
    # loss = -prob[torch.arange(len(X)), Y].log().mean()
    
    # Backward pass
    for p in params:
        p.grad = None
    loss.backward()
    
    # Update
    lr = 0.1 if i < 100000 else 0.01
    for p in params:
        p.data += -lr * p.grad

    # Track stats
    # lri.append(lre[i])
    # lossi.append(loss.log10().item())
    # stepi.append(i)
    
emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * feature_length) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

# plt.plot(stepi, lossi)
# plt.show()

for _ in range(10): # Write reviews
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(-1, block_size * feature_length) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print('-------------------------------------')
    print(decode(out))
