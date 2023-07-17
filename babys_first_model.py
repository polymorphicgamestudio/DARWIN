"""
Simple bigraph character level language model
NOTE(caleb): Bigraph language models apprently SUCK so don't expect great outputs...
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm # Progress bars

mc_path = 'data/clean_McDonald_s_Reviews.txt'
mc_reviews = open(mc_path, 'r').read().splitlines()

# Unique characters in data set
chars = sorted(list(set(''.join(mc_reviews))))
vocab_size = len(chars)
print(''.join(chars))

# Create a mapping from characters to integers
stoi = { s:i+1 for i,s in enumerate(chars) }
special = '🔥' # NOTE(caleb): Some unique identifier (that isn't in vocab)
stoi[special] = 0
vocab_size += 1
itos = { i:s for s,i in stoi.items() } # Map back to characters
encode = lambda s: [stoi[c] for c in s] # Take a string, output list of characters 
decode = lambda l: ''.join([itos[i] for i in l]) # Take a list of integers, output a string

# Build 'counts' matrix 
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for review in tqdm(mc_reviews):
    chs = [special] + list(review) + [special]
    for ch1, ch2 in zip(chs, chs[1:]):
        ch1_index = stoi[ch1]
        ch2_index = stoi[ch2]
        N[ch1_index, ch2_index] += 1

# Make this script deterministic
g = torch.Generator().manual_seed(2147483647) 

# Go ahead and precompute probability matrix (probabilty distribution)
P = (N+1).float() # N + 1 aka model smoothing
P /= P.sum(1, keepdim=True) # NOTE(caleb): Optimized eqv of:
# for row_index in range(vocab_size):
#     P[row_index] /= P[row_index].sum() 

# Create the training set of all the bigrams
xs, ys = [], []
for review in mc_reviews:
    chs = [special] + list(review) + [special]
    for ch1, ch2 in zip(chs, chs[1:]):
        ch1_index = stoi[ch1]
        ch2_index = stoi[ch2]
        xs.append(ch1_index)
        ys.append(ch2_index)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('Number of examples: ', num)

# Randomly initialize vocab_size neurons' weights. Each neuron receives vocab_size inputs
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

# Gradient descent
print("GRADIENT DESCENT (this might take a while):")
for k in tqdm(range(5)):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=vocab_size).float() # Input to the neural network: one_hot encoding
    logits = xenc @ W # Predict log-counts
    counts = logits.exp() # Something that looks like counts (N matrix)
    probs = counts / counts.sum(1, keepdims=True) # Probabilities for next character
    loss = -probs[torch.arange(len(xs)), ys].log().mean()
    # Backward pass
    W.grad = None # Set gradient to zero
    loss.backward()
    # Update
    W.data += -10 * W.grad
    
    # print(loss.item())

    
for i in range(10): # Write 10 reviews
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W # Predict log-counts
        counts = logits.exp() # Something that looks like counts (N matrix)
        p = counts / counts.sum(1, keepdims=True) # Probabilities for next character
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        else:
            out.append(itos[ix])
    print('-------------------------------------')
    print(''.join(out)) 

    
# Compute negative log likelihood 
# NOTE(caleb): This single number summarizes model quality (lower is better)
log_likelihood = 0.0
n = 0
for review in mc_reviews[:1]: # nll of this model producing the first review
    print(review)
    chs = [special] + list(review) + [special]
    for ch1, ch2 in zip(chs, chs[1:]):
        ch1_index = stoi[ch1]
        ch2_index = stoi[ch2]
        prob = P[ch1_index, ch2_index]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')         
