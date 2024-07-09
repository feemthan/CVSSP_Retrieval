import numpy as np
import sys

img_emb = np.load('multimodal_embeddings/img_emb/img_emb_0.npy')
text_emb = np.load('multimodal_embeddings/text_emb/text_emb_0.npy')

# Elementwise combination for multimodal embeddings
w1 = 0.5
w2 = 0.5

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

multimodal_emb = normalized(w1*img_emb + w2*text_emb)
np.save('multimodal_embeddings/combined_emb/combined_emb.npy', multimodal_emb)

