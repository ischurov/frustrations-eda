from pathlib import Path

import numpy as np

import os
from os import path

import matplotlib.pylab as plt
import matplotlib

import pickle
import time  



L_TYPE='kagome'
EPOCHS_N = 500
N_COMPONENTS=100
NET_SIZE=512

matplotlib.rcParams['font.family'] = "DejaVu Sans"

#j2=1.3_kagome_acc_1699619027.8651476.pickle
#j2=1.3_triangle_acc_1699619027.8651476.pickle


file = open('j2=1.3_kagome_acc_1699619027.8651476.pickle', 'rb')
kagome_acc=pickle.load(file)
file.close()

file = open('j2=1.3_triangle_acc_1699619027.8651476.pickle', 'rb')
triangle_acc=pickle.load(file)
file.close()

x=range(0,10000,100)

plt.clf()
plt.xlabel('terms', fontsize=20)
plt.plot(x,kagome_acc,color='orange', linewidth=2, label='kagome')
plt.plot(x,triangle_acc,color='blue', linewidth=2, label='triangle')
plt.ylim(0.45,1)
plt.ylabel('accuracy', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax = plt.subplot(111)
ax.spines[['right', 'top']].set_visible(False)


plt.title('$J_2=1.3$, 500 epochs', fontsize=20)
plt.legend(loc='upper right', fontsize=18)
plt.tight_layout()
plt.savefig(str(N_COMPONENTS)+'terms_comparison_'+str(EPOCHS_N)+'epochs.png')




### for gradual detruncating:
''''
step 99, [50, 422] loss: 0.4669061601161957
Train set: accuracy: 76.47058823529412 %, sign overlap: 0.39397498965263367
Validation set: accuracy: 67.61794479100519 %, sign overlap: 0.511311948299408
Original test set: accuracy: 62.70610874240652 %, sign overlap: 0.36083588004112244

'''

# for original set
'''
step 99, [50, 422] loss: 0.5728375315666199
Train set: accuracy: 76.47058823529412 %, sign overlap: 0.8639479279518127
Validation set: accuracy: 68.00642486272459 %, sign overlap: 0.5437372922897339
Original test set: accuracy: 65.47937969286495 %, sign overlap: 0.453641414642334
'''
