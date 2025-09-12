
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

import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14,'axes.titlesize': 15, 'figure.titlesize' : 16})

matplotlib.rcParams['font.family'] = "DejaVu Sans"
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['figure.titlesize'] = 20


import umap.umap_ as umap
reducer = umap.UMAP()

import sys

N_COMPONENTS=int(sys.argv[1])

#N_COMPONENTS=30000

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
	fig.savefig('umap/' + outname)
	plt.close()


def plot_umap(data, coeffs, l_type, j2_n):
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
	l_type_trimmed=l_type.split('_')[0]
	embedding = reducer.transform(data)
	# Verify that the result of calling transform is
	# idenitical to accessing the embedding_ attribute
	assert(np.all(embedding == reducer.embedding_))
	print(embedding.shape)
	alphas=(np.abs(coeffs)/np.max(np.abs(coeffs)))
	print(coeffs)
	plt.clf()
	plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18,'axes.titlesize': 20, 'figure.titlesize' : 20})
	#plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14,'axes.titlesize': 15, 'figure.titlesize' : 16})
	plt.scatter(embedding[:, 0], embedding[:, 1], c=np.sign(coeffs), cmap='Spectral', s=2, alpha=alphas) #FIXME!!!!!!!!!!!!!!! alpha=alphas, 
	plt.gca().set_aspect('equal', 'datalim')
	#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
	
	plt.title('Boolean Fourier transform, '+l_type_trimmed+', $J_2='+str(j2_n)+'$')#, fontsize=12);

	plt.savefig('/umap/umap_fourier_'+str(N_COMPONENTS)+'_'+l_type+'.png')


triangle_lattice = TriangleLattice(6, 4)
kagome_lattice=KagomeLattice(2,4)
square_lattice=SquareLattice(6,4)

print('generated lattices')
'''
system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=1.25, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series = fourier_expand(lbf)
'''

##############

system = HeisenbergJ1J2(kagome_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
kagome_series = fourier_expand(lbf)


system = HeisenbergJ1J2(square_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
square_series = fourier_expand(lbf)

###################
system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_1 = fourier_expand(lbf)

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=1.25, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_125 = fourier_expand(lbf)


system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.8, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_08 = fourier_expand(lbf)

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.4, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_04 = fourier_expand(lbf)

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.9, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_09 = fourier_expand(lbf)

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.91, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_091 = fourier_expand(lbf)

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.92, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_092 = fourier_expand(lbf)


system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.93, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_093 = fourier_expand(lbf)

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.94, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_094 = fourier_expand(lbf)


system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=0.95, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series_095 = fourier_expand(lbf)






triangle_1_largest_indices, triangle_1_largest_coeffs = get_abslargest_terms(triangle_series_1.coeffs, N_COMPONENTS)
triangle_08_largest_indices, triangle_08_largest_coeffs = get_abslargest_terms(triangle_series_08.coeffs, N_COMPONENTS)
triangle_04_largest_indices, triangle_04_largest_coeffs = get_abslargest_terms(triangle_series_04.coeffs, N_COMPONENTS)
triangle_09_largest_indices, triangle_09_largest_coeffs = get_abslargest_terms(triangle_series_09.coeffs, N_COMPONENTS)

triangle_091_largest_indices, triangle_091_largest_coeffs = get_abslargest_terms(triangle_series_091.coeffs, N_COMPONENTS)
triangle_092_largest_indices, triangle_092_largest_coeffs = get_abslargest_terms(triangle_series_092.coeffs, N_COMPONENTS)
triangle_093_largest_indices, triangle_093_largest_coeffs = get_abslargest_terms(triangle_series_093.coeffs, N_COMPONENTS)
triangle_094_largest_indices, triangle_094_largest_coeffs = get_abslargest_terms(triangle_series_094.coeffs, N_COMPONENTS)


triangle_095_largest_indices, triangle_095_largest_coeffs = get_abslargest_terms(triangle_series_095.coeffs, N_COMPONENTS)
triangle_125_largest_indices, triangle_125_largest_coeffs = get_abslargest_terms(triangle_series_125.coeffs, N_COMPONENTS)

kagome_largest_indices, kagome_largest_coeffs = get_abslargest_terms(kagome_series.coeffs, N_COMPONENTS)
square_largest_indices, square_largest_coeffs = get_abslargest_terms(square_series.coeffs, N_COMPONENTS)

triangle_1_xors=make_unpacked_configurations(triangle_1_largest_indices,24)
triangle_08_xors=make_unpacked_configurations(triangle_08_largest_indices,24)
triangle_04_xors=make_unpacked_configurations(triangle_04_largest_indices,24)
triangle_09_xors=make_unpacked_configurations(triangle_09_largest_indices,24)

triangle_091_xors=make_unpacked_configurations(triangle_091_largest_indices,24)
triangle_092_xors=make_unpacked_configurations(triangle_092_largest_indices,24)
triangle_093_xors=make_unpacked_configurations(triangle_093_largest_indices,24)
triangle_094_xors=make_unpacked_configurations(triangle_094_largest_indices,24)

triangle_095_xors=make_unpacked_configurations(triangle_095_largest_indices,24)
triangle_125_xors=make_unpacked_configurations(triangle_125_largest_indices,24)

kagome_xors=make_unpacked_configurations(kagome_largest_indices,24)
square_xors=make_unpacked_configurations(square_largest_indices,24)

'''
a=triangle_xors.T*np.abs(triangle_largest_coeffs.flatten())
triangle_xors_weighted=a.T
a=kagome_xors.T*np.abs(kagome_largest_coeffs.flatten())
kagome_xors_weighted=a.T
a=square_xors.T*np.abs(square_largest_coeffs.flatten())
square_xors_weighted=a.T
'''

print('fourier transform finished')




plot_umap(triangle_1_xors, triangle_1_largest_coeffs, 'triangle_j2=1', 1)
plot_umap(triangle_08_xors, triangle_08_largest_coeffs, 'triangle_j2=08', 0.8)
plot_umap(triangle_04_xors, triangle_04_largest_coeffs, 'triangle_j2=04', 0.4)
plot_umap(triangle_09_xors, triangle_09_largest_coeffs, 'triangle_j2=09', 0.9)

plot_umap(triangle_091_xors, triangle_091_largest_coeffs, 'triangle_j2=091', 0.91)
plot_umap(triangle_092_xors, triangle_092_largest_coeffs, 'triangle_j2=092', 0.92)
plot_umap(triangle_093_xors, triangle_093_largest_coeffs, 'triangle_j2=093', 0.93)
plot_umap(triangle_094_xors, triangle_094_largest_coeffs, 'triangle_j2=094', 0.94)

plot_umap(triangle_095_xors, triangle_095_largest_coeffs, 'triangle_j2=095', 0.95)
plot_umap(triangle_125_xors, triangle_125_largest_coeffs, 'triangle_j2=125', 0.96)

plot_umap(square_xors, square_largest_coeffs, 'square', 1)
plot_umap(kagome_xors, kagome_largest_coeffs, 'kagome', 1)


