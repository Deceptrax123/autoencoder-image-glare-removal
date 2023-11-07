# Deep Learning Based Glare Removal from Images
This repository is a part of <a href="https://github.com/Deceptrax123/Negative-Transfer-Mitigation">Negative-Transfer-Mitigation</a>. It contains the code for
a base autoencoder model used to learn the distribution of images without glare and generate the same given an input with many glare spots. 

Two Models were trained:
1. Model to reduce brightness to a specific level
2. Model to reduce glare through image inpainting. 

The models have been written using PyTorch. 