# PCA-for-Face-Recognition
Required Python Library: OpenCV, Numpy, PIL
- Place the program (eface.py) in the same folder with images directory.
- Computer the program by running:
```
python eface.py --pc NUM
```
where NUM is the number of component to be executing. NUM default value is 100.

After running, the program will output: 
- runtime for computing PCA
- runtime and score for comparing all PCA of same face
- runtime and score for comparing all PCA of different face

The score will then be used for plotting in R, and also for calculate FMNR and FMR.


