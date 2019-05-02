import numpy as np
import cv2
import glob
import time
import argparse
from PIL import Image
parser = argparse.ArgumentParser(description='PCA eigenface')
parser.add_argument('-c', '--pc', type=int, default=100,\
                    help='number of principle component')
args = parser.parse_args()

def to_np_array(images):
    np_images = np.zeros((len(images),images[0].size[0]*images[0].size[1]))
    for i in range(len(images)) :
        np_images[i] = (np.array(list(images[i].getdata())))
    return np_images

def compare_same(mean,evec):
    start = time.time()
    score = []
    image_list = []

    for filename in sorted(glob.glob("images/Test/*.jpg")): 
        im=Image.open(filename)
        image_list.append(im)
    a = []
    for i in image_list:
        a.append(i.resize((64, 64), Image.ANTIALIAS))
    test_data = to_np_array(a)
    
    for i in range(int(len(test_data)/2)):  
        img1 = test_data[i*2]-mean
        img2 = test_data[(i*2)+1]-mean
        
        i1 = np.dot(img1, evec)
        i2 = np.dot(img2, evec)
        diff = i2-i1
        norms = np.linalg.norm(diff, axis=0)
        #score.append(norms)
        #s = np.argmin(norms).item()
        score.append(norms)

    print("compare same")
    print(time.time() - start)
    print(score)
    score.sort()
    # hmean = np.mean(score)
    # hstd = np.std(score)
    # pdf = stats.norm.pdf(score, hmean, hstd)
    

def compare_diff(mean, evec):
    start = time.time()
    #test_data = imread_collection("images/Test/*.jpg")
    score = []
    image_list = []
    k=0
    l=0
    for filename in sorted(glob.glob("images/Test/*.jpg")): 
        im=Image.open(filename)
        image_list.append(im)
    a = []
    for i in image_list:
        a.append(i.resize((64, 64), Image.ANTIALIAS))
    test_data = to_np_array(a)

    for i in range(int(len(test_data)/2)):
        comp = []
        
        for j in range(int(len(test_data)/2)):
            if i==j:
                continue  

            img1 = test_data[i*2]-mean
            img2 = test_data[j*2]-mean
            i1 = np.dot(img1, evec)
            i2 = np.dot(img2, evec)
            diff = i1-i2
            norms = np.linalg.norm(diff, axis=0)
            s = np.argmin(norms).item()
            
            #s = np.argmin(norms).item()
            #b.append(s)
            comp.append(norms)
            score.append(norms)
    print("compare diff")
    print(time.time() - start)
    print(score)
    


#train_data = imread_collection("images/Training/*.jpg")
PC_NUM = args.pc
start = time.time()
image_list = []
for filename in glob.glob("images/Training/*.jpg"): #assuming gif
    im=Image.open(filename)
    image_list.append(im)
a = []
for i in image_list:
    a.append(i.resize((64, 64), Image.ANTIALIAS))
data = to_np_array(a)

mean = data.mean(axis=0)

data = (data-mean)
c = np.cov(data, rowvar=False)
eig_val, eig_vec = np.linalg.eigh(c)
idx = eig_val.argsort()[::-1]
eig_val = eig_val[idx]
eig_vec = eig_vec[:,idx]
eig_vec = eig_vec[:, :PC_NUM]
print("done computing PCA")
print(time.time() - start)

# transformed = np.dot(eig_vec.T, data[0].T).T

# t = transformed.dot(eig_vec.T) + mean

compare_same(mean, eig_vec)
compare_diff(mean, eig_vec)

# plt.imshow(t.reshape(64,64), cmap=plt.cm.bone)
# plt.show()
# print(data.shape)


