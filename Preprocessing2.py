import os
import numpy as np

raw_npy_dir = '/Users/apple/Downloads/Pop_npy/'
npy_phrases_dir = '/Users/apple/Downloads/Pop_phrases/'

if not os.path.exists(npy_phrases_dir):
    os.makedirs(npy_phrases_dir)

# concatenate into a long npy file, then crop into four-bar numpy phrases
l = [f for f in os.listdir(raw_npy_dir)]
print(l)
train = np.load(os.path.join(raw_npy_dir, l[0]))
print(train.shape, np.max(train))
for i in range(1, len(l)):
    try:
        print(i, l[i])
        t = np.load(os.path.join(raw_npy_dir, l[i])) 
        # use 'try&except' to ignore the .DS_store file
    except:
        continue
    train = np.concatenate((train, t), axis=0)
print(train.shape)
np.save('/Users/apple/Downloads/concatenated_long.npy', (train > 0.0))
#train = train > 0.0

# crop into 4-bar numpy phrases
if not os.path.exists(npy_phrases_dir):
    os.makedirs(npy_phrases_dir)
x = np.load('/Users/apple/Downloads/concatenated_long.npy')
print(x.shape)
count = 0
for i in range(x.shape[0]):
    if np.max(x[i]):
        count += 1
        np.save(os.path.join(npy_phrases_dir, 'Pop_{}.npy'.format(i+1)), x[i])
        print(x[i].shape)
    if count == 20000:
        break
print("{} npy phrases generated".format(count))
os.remove('/Users/apple/Downloads/concatenated_long.npy')
