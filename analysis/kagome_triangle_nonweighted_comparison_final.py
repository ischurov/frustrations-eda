from pathlib import Path

import numpy as np

from fast_boolean_analysis import fourier_expand, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem, SignSignalKind
from spin_lattices import KagomeLattice, TriangleLattice

import os
from os import path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import matplotlib.pylab as plt

import pandas as pd

import pickle
import time  



L_TYPE='kagome'
EPOCHS_N = 500
N_COMPONENTS=100
NET_SIZE=512

#this one should use a larger more expressive network and uniform sampling

def get_inputs_and_labels(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(df[features].values.astype("float"), dtype=torch.float32)
    y = torch.tensor(df["y"].values.astype("int8"), dtype=torch.long)
    probs = torch.tensor(df["prob"].values.astype("float"), dtype=torch.float32)
    return X, y, probs



def evaluate(net, inputs, labels, probs):
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)

        sign_overlap = (
            ((predicted * 2 - 1) * (labels * 2 - 1) * probs).sum() / probs.sum()
        ).item()
        return accuracy, sign_overlap


triangle_lattice = TriangleLattice(6, 4)
kagome_lattice=KagomeLattice(2,4)

lattices=[triangle_lattice, kagome_lattice]
j2_i=1.3
kagome_acc=[]
triangle_acc=[]
components_all=np.arange(100,N_COMPONENTS+10,100)

lattice_i=0
for lattice in lattices:
	system = HeisenbergJ1J2(lattice, J1=1, J2=j2_i, ground_state_cache_dir=Path("groundstates"))

	lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
	series = fourier_expand(lbf)
	ground_state = system.get_ground_state_in_canonical_basis()

	df = (system.get_df_ground_state(
			canonical_basis=True, unpack_configurations=True, expand_basis_columns=True
		    )
		    .assign(
			sign=(lambda df: np.sign(df["eigenstate_coeff"])),
			prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
		    )
		    .assign(y=lambda df: (df["sign"] == 1).astype(int))
		)

	df_original=df.copy()
	assert (series.predict() == np.sign(ground_state)).all()
	for N_COMPONENTS in components_all:
		# for untrunctaed series, prediction of the signal is perfect

		net = nn.Sequential(
		    nn.Linear(system.number_spins, 512),
		    nn.ReLU(),
		    nn.Linear(512, 2),
		)


		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
			
		df=df_original.copy()



		features = [f"s{i}" for i in range(system.number_spins)]


		eps_train = 1e-2
		val_eps = 1e-2
		test_eps = 1e-2
		batch_size = 64
		


		s1=series.truncate(keep_largest_n(N_COMPONENTS))
		s1.coeffs=np.sign(s1.coeffs)

		prediction = s1.predict()
		ps=np.sign(prediction)
		
		df=df_original.copy()
		df['sign']=ps
		df=df.assign(y=lambda df: (df['sign'] == 1).astype(int))


		df_train = df.sample(frac=eps_train, weights="prob")
		df_val = df.drop(df_train.index).sample(frac=val_eps, weights="prob")		
		inputs_val, labels_val, probs_val = get_inputs_and_labels(df_val)

		'''
		data_test = df.drop(df_train.index).drop(df_val.index).sample(frac=test_eps, weights="prob") #should be a uniform distribution if you remove it
		inputs_test, labels_test, probs_test = get_inputs_and_labels(data_test)
		'''


		n_batches = int(np.ceil(len(df_train) / batch_size))
		epochs = EPOCHS_N #20000



		

		#dynamics of learning the original set
		acc=[]

		#dynamics of learning the truncated set
		truncated_acc=[]
		truncated_acc_orig=[]

		#dynamics of learning the truncated set with only sign left of coefficients
		sign_acc=[]
		sign_acc_orig=[]

		loss_trunc=[]

		for epoch in range(epochs):  
				running_loss = 0.0
				i = None
				loss = None

				for i in range(n_batches):
					data = df_train.iloc[i * batch_size : (i + 1) * batch_size]
					inputs, labels, probs = get_inputs_and_labels(data)

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward + backward + optimize
					outputs = net(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
				final_loss=loss.detach().tolist()
				print("loss: ", str(final_loss))
				loss_trunc.append(final_loss)

					#print(f"step {seg}, [{epoch + 1}, {i}] loss: {loss}")
				accuracy_val, sign_overlap_val = evaluate(net, inputs_val, labels_val, probs_val)
				print(f"Validation set: accuracy: {100 * accuracy_val} %, sign overlap: {sign_overlap_val}")

		if lattice_i:
			kagome_acc.append(accuracy_val)
		else:
			triangle_acc.append(accuracy_val)#FIXME



	x=components_all

time_stamp = time.time()



file = open('j2='+str(j2_i)+'_all_acc_'+str(time_stamp)+'.pickle', 'wb')
pickle.dump(triangle_acc, file)
file.close()


kagome_acc=triangle_acc[100:200] #this is a hotfix. FIXME
triangle_acc=triangle_acc[0:100]


file = open('j2='+str(j2_i)+'_kagome_acc_'+str(time_stamp)+'.pickle', 'wb')
pickle.dump(kagome_acc, file)
file.close()

file = open('j2='+str(j2_i)+'_triangle_acc_'+str(time_stamp)+'.pickle', 'wb')
pickle.dump(triangle_acc, file)
file.close()

plt.clf()
plt.xlabel('terms')
plt.plot(x,kagome_acc,color='orange', linewidth=2, label='kagome')
plt.plot(x,triangle_acc,color='blue', linewidth=2, label='triangle')
plt.ylim(0.45,1)
plt.ylabel('accuracy')
plt.title('comparison of kagome and triangle at j2=' +str(j2_i)+ ', '+str(EPOCHS_N)+' epochs')
plt.legend(loc='lower right')
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
