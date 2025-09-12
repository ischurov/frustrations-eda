
from pathlib import Path

import numpy as np

from fast_boolean_analysis import fourier_expand, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem, SignSignalKind
from spin_lattices import KagomeLattice, TriangleLattice, SquareLattice

import os
from os import path

import numba

from scipy.spatial import distance


import matplotlib.patches as mpatches

import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns 

import pickle

from utils import make_unpacked_configurations
from utils import get_abslargest_terms


from sklearn.preprocessing import StandardScaler

import itertools


import umap.umap_ as umap
reducer = umap.UMAP()


N_COMPONENTS=1000



@numba.njit()
def flip_dist(a,b):
    return (np.sum(a)-np.sum(b)) % 2


def rdm(act_matrix, coeffs):
	rdm_matrix = distance.squareform(distance.pdist(act_matrix,metric='hamming'))
	return rdm_matrix	

def rdm_weighted(act_matrix, coeffs):
	corr_vector=[]
	for i in range(0,len(act_matrix)):
		for j in range(i, len(act_matrix)):
			#calculated correlation+weights 
			corr_vector.append(distance.hamming(act_matrix[i], act_matrix[j])*coeffs[i]*coeffs[j])
	return distance.squareform(corr_vector)



def rdm_and(act_matrix, coeffs):
	corr_vector=[]
	for i in range(0,len(act_matrix)):
		for j in range(i, len(act_matrix)):
			#calculated correlation+weights 
			corr_vector.append( np.abs(6-np.sum(np.multiply(act_matrix[i], act_matrix[j]))) *coeffs[i]*coeffs[j])
			#corr_vector.append(np.sum(np.multiply(act_matrix[i], act_matrix[j]))*coeffs[i]*coeffs[j])
	return distance.squareform(corr_vector)


def corr_weighted(act_matrix, coeffs): #FIXME: combine with the previous two
	corr_vector=[]
	for i in range(0,len(act_matrix)):
		for j in range(i, len(act_matrix)):
			#calculated correlation+weights 
			corr_vector.append(distance.hamming(act_matrix[i], act_matrix[j])*coeffs[i]*coeffs[j])
	return corr_vector

def corr(act_matrix, coeffs): #FIXME: combine with the previous two
	corr_vector=[]
	for i in range(0,len(act_matrix)):
		for j in range(i, len(act_matrix)):
			#calculated correlation+weights 
			corr_vector.append(distance.hamming(act_matrix[i], act_matrix[j]))
	return corr_vector


def corr_and(act_matrix, coeffs): #FIXME: combine with the previous two
	corr_vector=[]
	for i in range(0,len(act_matrix)):
		for j in range(i, len(act_matrix)):
			#calculated correlation+weights 
			corr_vector.append(np.abs(6-np.sum(np.multiply(act_matrix[i], act_matrix[j])))*coeffs[i]*coeffs[j])
	return corr_vector

def rdm_plot(rdm, vmin, vmax, labels, outname):
	fig, ax = plt.subplots(figsize=(20,20))
	sns.heatmap(rdm, ax=ax,  vmin=vmin, vmax=vmax, xticklabels=labels, yticklabels=labels) #cmap='Blues_r',
	fig.savefig('outname)
	plt.close()


def plot_umap(data, coeffs, amplitudes, l_type):
	reducer=umap.UMAP(a=None, angular_rp_forest=False, b=None,
	     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
	     local_connectivity=1.0, low_memory=False, metric=flip_dist,
	     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
	     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
	     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
	     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
	     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
	     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
	reducer.fit(data)

	'''
	green_patch = mpatches.Patch(color='green', label='long neck and ears or short neck and ears')
	#blue_patch = mpatches.Patch(color='blue', label='short neck and ears')
	orange_patch = mpatches.Patch(color='orange', label='random')
	'''

	embedding = reducer.transform(data)
	# Verify that the result of calling transform is
	# idenitical to accessing the embedding_ attribute
	assert(np.all(embedding == reducer.embedding_))
	print(embedding.shape)
	alphas=(np.abs(amplitudes)/np.max(np.abs(amplitudes)))
	#print(coeffs)
	plt.clf()
	plt.scatter(embedding[:, 0], embedding[:, 1], c=np.sign(coeffs), cmap='Spectral', s=2, alpha=alphas) #FIXME!!!!!!!!!!!!!!! alpha=alphas, 
	plt.gca().set_aspect('equal', 'datalim')
	#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
	plt.title('UMAP projection, '+l_type+' '+str(N_COMPONENTS)+' components', fontsize=12);

	plt.savefig('umap/umap_lattice_'+str(N_COMPONENTS)+'_'+l_type+'.png')

triangle_lattice = TriangleLattice(6, 4)
kagome_lattice=KagomeLattice(2,4)
square_lattice=SquareLattice(6,4)

print('generated lattices')

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.8, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series = fourier_expand(lbf)

triangle_df = (system.get_df_ground_state(
			canonical_basis=True, unpack_configurations=True, expand_basis_columns=True
		    )
		    .assign(
			sign=(lambda df: np.sign(df["eigenstate_coeff"])),
			prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
		    )
		    .assign(y=lambda df: (df["sign"] == 1).astype(int))
		)

system = HeisenbergJ1J2(kagome_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
kagome_series = fourier_expand(lbf)

kagome_df = (system.get_df_ground_state(
			canonical_basis=True, unpack_configurations=True, expand_basis_columns=True
		    )
		    .assign(
			sign=(lambda df: np.sign(df["eigenstate_coeff"])),
			prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
		    )
		    .assign(y=lambda df: (df["sign"] == 1).astype(int))
		)

system = HeisenbergJ1J2(square_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
square_series = fourier_expand(lbf)


square_df = (system.get_df_ground_state(
			canonical_basis=True, unpack_configurations=True, expand_basis_columns=True
		    )
		    .assign(
			sign=(lambda df: np.sign(df["eigenstate_coeff"])),
			prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
		    )
		    .assign(y=lambda df: (df["sign"] == 1).astype(int))
		)


triangle_largest_indices, triangle_largest_coeffs = get_abslargest_terms(triangle_series.coeffs, N_COMPONENTS)
kagome_largest_indices, kagome_largest_coeffs = get_abslargest_terms(kagome_series.coeffs, N_COMPONENTS)
square_largest_indices, square_largest_coeffs = get_abslargest_terms(square_series.coeffs, N_COMPONENTS)

triangle_xors=make_unpacked_configurations(triangle_largest_indices,24)
kagome_xors=make_unpacked_configurations(kagome_largest_indices,24)
square_xors=make_unpacked_configurations(square_largest_indices,24)

a=triangle_xors.T*np.abs(triangle_largest_coeffs.flatten())
triangle_xors_weighted=a.T
a=kagome_xors.T*np.abs(kagome_largest_coeffs.flatten())
kagome_xors_weighted=a.T
a=square_xors.T*np.abs(square_largest_coeffs.flatten())
square_xors_weighted=a.T

print('fourier transform finished')


features = [f"s{i}" for i in range(system.number_spins)]



kagome_head=kagome_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')
triangle_head=triangle_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')
square_head=square_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')

plot_umap(kagome_head[features].values, kagome_head['sign'].values, kagome_head['amplitude'].values, 'kagome')
plot_umap(triangle_head[features].values, triangle_head['sign'].values,  triangle_head['amplitude'].values, 'triangle')
plot_umap(square_head[features].values, square_head['sign'].values,  square_head['amplitude'].values, 'square')


################DSI###########################################

'''

triangle_head=triangle_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')


labels=triangle_head['sign'].values
npdata=triangle_head[features].values

npdata=npdata.astype(int)


file = open('triangle_lattice_xors_'+str(N_COMPONENTS)+'.pickle', 'wb')
pickle.dump(npdata, file)
file.close()

file = open('triangle_lattice_signs_'+str(N_COMPONENTS)+'.pickle', 'wb')
pickle.dump(labels, file)
file.close()


class_a=max(labels)
class_b=min(labels)

class_a_indices=np.where(labels==class_a)
class_a_values=npdata[class_a_indices]
class_b_indices=np.where(labels==class_b)
class_b_values=npdata[class_b_indices]

set1 = class_a_values
set2 = class_b_values

intra_1 = []

for el11 in set1:
    for el12 in set1:    
        intra_1.append(np.sum(np.abs(el11-el12)))

intra_2 = []

for el21 in set2:
    for el22 in set2:    
        intra_2.append(np.sum(np.abs(el21-el22)))
        
inter = []
        
for el1 in set1:
    for el2 in set2:    
        inter.append(np.sum(np.abs(el1-el2)))
        
print(intra_1)

minimal = min(min(intra_1),min(intra_2),min(inter))
maximal = max(max(intra_1),max(intra_2),max(inter))

print(minimal, min(intra_1), min(intra_2), min(inter))
print(maximal, max(intra_1), max(intra_2), max(inter))

dist1 = scipy.stats.kstest(intra_1, inter, args=(), N=20, alternative='two-sided')
print(dist1)
dist2 = scipy.stats.kstest(intra_2, inter, args=(), N=20, alternative='two-sided')
print(dist2)
dist3 = scipy.stats.kstest(intra_1, intra_2, args=(), N=20, alternative='two-sided')
print(dist3)

print("Separability Index = ", (dist1[0] + dist2[0])/2)


# fixed bin size

plt.clf()
bins = np.arange(minimal, maximal, 1) # fixed bin size

plt.xlim([minimal, maximal])

plt.hist(intra_1, bins=bins, alpha=0.5)
plt.hist(intra_2, bins=bins, alpha=0.5)
plt.hist(inter, bins=bins, alpha=0.5)
plt.title('Distances histogram')
plt.xlabel('distance')
plt.ylabel('count')

plt.savefig('triangle_lattice_hist_dsi'+str(N_COMPONENTS)+'.png')
'''



