
from pathlib import Path

import numpy as np

from fast_boolean_analysis import fourier_expand, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem, SignSignalKind
from spin_lattices import KagomeLattice, TriangleLattice, SquareLattice

import os
from os import path


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


N_COMPONENTS=30000

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
	fig.savefig('/vol/tcm11/kravchenko/' + outname)
	plt.close()


def plot_umap(data, coeffs, l_type):
	reducer=umap.UMAP(a=None, angular_rp_forest=False, b=None,
	     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
	     local_connectivity=1.0, low_memory=False, metric='euclidean',
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
	alphas=(np.abs(coeffs)/np.max(np.abs(coeffs)))
	print(coeffs)
	plt.clf()
	plt.scatter(embedding[:, 0], embedding[:, 1], c=np.sign(coeffs), cmap='Spectral', s=2) #FIXME!!!!!!!!!!!!!!! alpha=alphas, 
	plt.gca().set_aspect('equal', 'datalim')
	#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
	plt.title('UMAP projection, '+l_type+' '+str(N_COMPONENTS)+' components', fontsize=12);

	plt.savefig('/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/results_polished_umap/umap_fourier_'+str(N_COMPONENTS)+'_'+l_type+'.png')

triangle_lattice = TriangleLattice(6, 4)
kagome_lattice=KagomeLattice(2,4)
square_lattice=SquareLattice(6,4)

print('generated lattices')

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=1.25, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series = fourier_expand(lbf)


system = HeisenbergJ1J2(kagome_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
kagome_series = fourier_expand(lbf)


system = HeisenbergJ1J2(square_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
square_series = fourier_expand(lbf)



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



plot_umap(kagome_xors, kagome_largest_coeffs, 'kagome')
plot_umap(triangle_xors, triangle_largest_coeffs, 'triangle')
plot_umap(square_xors, square_largest_coeffs, 'square')



'''
reducer=umap.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
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

plt.scatter(embedding[:, 0], embedding[:, 1], c=np.sign(kagome_largest_coeffs), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection', fontsize=24);
plt.title('kagome, '+str(N_COMPONENTS)+' components, UMAP projection of the dataset', fontsize=24);

plt.savefig('umap_test.png')


'''


######################
'''
a=triangle_xors.T*np.abs(triangle_largest_coeffs.flatten())
triangle_xors_weighted=a.T
a=kagome_xors.T*np.abs(kagome_largest_coeffs.flatten())
kagome_xors_weighted=a.T
a=square_xors.T*np.abs(square_largest_coeffs.flatten())
square_xors_weighted=a.T
'''

'''
####weighted version
triangle_xors_intersection=np.sum(triangle_xors_weighted, axis=0)
kagome_xors_intersection=np.sum(kagome_xors_weighted, axis=0)
square_xors_intersection=np.sum(square_xors_weighted, axis=0)


triangle_rdm=rdm(triangle_xors_weighted)
kagome_rdm=rdm(kagome_xors_weighted)
square_rdm=rdm(square_xors_weighted)
######################

'''
'''
triangle_xors_intersection=np.sum(triangle_xors, axis=0)
kagome_xors_intersection=np.sum(kagome_xors, axis=0)
square_xors_intersection=np.sum(square_xors, axis=0)


triangle_rdm=rdm_and(triangle_xors, np.abs(triangle_largest_coeffs)/np.sum(np.abs(triangle_largest_coeffs)))
kagome_rdm=rdm_and(kagome_xors, np.abs(kagome_largest_coeffs)/np.sum(np.abs(kagome_largest_coeffs)))
square_rdm=rdm_and(square_xors, np.abs(square_largest_coeffs)/np.sum(np.abs(square_largest_coeffs)))

lbl=list(range(0,triangle_xors.shape[0]))
rdm_plot(triangle_rdm, vmin = np.min(triangle_rdm), vmax = np.max(triangle_rdm), labels = lbl, outname = 'triangle_rdm_and_'+str(N_COMPONENTS)+'.png')
rdm_plot(kagome_rdm, vmin = np.min(kagome_rdm), vmax = np.max(kagome_rdm), labels = lbl, outname = 'kagome_rdm_and_'+str(N_COMPONENTS)+'.png')
rdm_plot(square_rdm, vmin = np.min(square_rdm), vmax = np.max(square_rdm), labels = lbl, outname = 'square_rdm_and_'+str(N_COMPONENTS)+'.png')

triangle_corr=corr_weighted(triangle_xors, np.abs(triangle_largest_coeffs)/np.sum(np.abs(triangle_largest_coeffs)))
kagome_corr=corr_weighted(kagome_xors, np.abs(kagome_largest_coeffs)/np.sum(np.abs(kagome_largest_coeffs)))
square_corr=corr_weighted(square_xors, np.abs(square_largest_coeffs)/np.sum(np.abs(square_largest_coeffs)))

x=range(0,len(triangle_corr))

a=np.sort(triangle_corr)
plt.clf()
plt.plot(x,a, linewidth="3")
plt.title('triangular lattice')
plt.savefig('/vol/tcm11/kravchenko/triangle_corr_weighted_hamming_'+str(N_COMPONENTS)+'.png')


a=np.sort(square_corr)
plt.clf()
plt.plot(x,a, linewidth="3")
plt.title('square lattice')
plt.savefig('/vol/tcm11/kravchenko/square_corr_weighted_hamming_'+str(N_COMPONENTS)+'.png')


b=np.sort(kagome_corr)
plt.clf()
plt.plot(x,b, linewidth="3")
plt.title('kagome lattice')
plt.savefig('/vol/tcm11/kravchenko/kagome_corr_weighted_hamming_'+str(N_COMPONENTS)+'.png')



plt.clf()
plt.yscale('log')
plt.plot(x,a,color='orange', label='triangular', linewidth="3")
plt.plot(x,b,color='blue', label='kagome', linewidth="3")
plt.title('50 components, not weighted, \'and\' mask')
plt.legend()
plt.savefig('/vol/tcm11/kravchenko/triangle_vs_kagome_'+str(N_COMPONENTS)+'_weighted_and.png')

'''
'''
количество ксоров в которых участвует каждый спин для треугольника:
[ 4,  4, 48,  4,  4, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
для кагоме:
[84, 68, 68, 32, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68]
для квадрата:
[0,   0, 100,   0,   2,  98,  98, 100,  98, 100,  98,  98,  98, 98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98]
'''
'''
>>> np.sum(kagome_xors_intersection)
1612
>>> np.mean(kagome_xors_intersection)
67.16666666666667

>>> np.sum(triangle_xors_intersection)
1888
>>> np.mean(triangle_xors_intersection)
78.66666666666667
>>> 
'''

'''
triangle_coef_sort=np.sort(np.abs(triangle_series.coeffs))
triangle_coef_truncated=triangle_coef_sort[-1000:]

kagome_coef_sort=np.sort(np.abs(kagome_series.coeffs))
kagome_coef_truncated=kagome_coef_sort[-1000:]


kagome_normalized=kagome_coef_truncated/np.sum(kagome_coef_truncated)
triangle_normalized=triangle_coef_truncated/np.sum(triangle_coef_truncated)
'''

'''
plt.clf()

x=list(range(0,1000))
plt.clf()
plt.bar(x, triangle_normalized, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('triangular lattice')
plt.savefig('triangle_coeffs.png')



plt.clf()
plt.bar(x, kagome_normalized, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('kagome lattice')
plt.savefig('kagome_coeffs.png')
'''


'''
for 10 components:

>>> exec(open("combine_images.py").read())
>>> np.sum(triangle_rdm)
2.3800000000000003
>>> np.sum(kagome_rdm)
3.640000000000001


'''



