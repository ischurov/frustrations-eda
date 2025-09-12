
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

import matplotlib
#matplotlib.rcParams['text.usetex'] = True

matplotlib.rcParams['font.family'] = "DejaVu Sans"
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['figure.titlesize'] = 20

#plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14,'axes.titlesize': 15, 'figure.titlesize' : 16})


from sklearn.preprocessing import StandardScaler

import itertools


import umap.umap_ as umap
reducer = umap.UMAP()


N_COMPONENTS=30000

import sys


N_COMPONENTS=int(sys.argv[1])



@numba.njit()
def flip_dist(a,b):
    return (np.sum(a)-np.sum(b)) % 2




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


def plot_umap(data, coeffs, amplitudes, l_type, j2_n):
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
	l_type_trimmed=l_type.split('_')[0]
	print(l_type_trimmed)

	embedding = reducer.transform(data)
	# Verify that the result of calling transform is
	# idenitical to accessing the embedding_ attribute
	assert(np.all(embedding == reducer.embedding_))
	print(embedding.shape)
	alphas=(np.abs(amplitudes)/np.max(np.abs(amplitudes)))
	#print(coeffs)
	plt.clf()
	plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18,'axes.titlesize': 20, 'figure.titlesize' : 20})
	plt.scatter(embedding[:, 0], embedding[:, 1], c=np.sign(coeffs), cmap='Spectral', s=2, alpha=alphas) #FIXME!!!!!!!!!!!!!!! alpha=alphas, 
	plt.gca().set_aspect('equal', 'datalim')
	#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
	plt.title('original sign structure, '+l_type_trimmed+', $J_2='+str(j2_n)+'$')#, fontsize=12);

	plt.savefig('umap/umap_lattice_'+str(N_COMPONENTS)+'_'+l_type+'.png')

triangle_lattice = TriangleLattice(6, 4)
kagome_lattice=KagomeLattice(2,4)
square_lattice=SquareLattice(6,4)

print('generated lattices')


#j2_array=[0.4,0.8,0.9,0.91,0.92,0.93,0.94,0.95,1,1.25]
triangle_systems=[]

j2_array=[0.8,0.9,0.91,0.92,0.93,0.94,0.95,1]

for j2_i in j2_array:
	system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=j2_i, ground_state_cache_dir=Path("groundstates"))
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
	triangle_systems.append(triangle_df)

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


#triangle_largest_indices, triangle_largest_coeffs = get_abslargest_terms(triangle_series.coeffs, N_COMPONENTS)



kagome_largest_indices, kagome_largest_coeffs = get_abslargest_terms(kagome_series.coeffs, N_COMPONENTS)
square_largest_indices, square_largest_coeffs = get_abslargest_terms(square_series.coeffs, N_COMPONENTS)

#triangle_xors=make_unpacked_configurations(triangle_largest_indices,24)


kagome_xors=make_unpacked_configurations(kagome_largest_indices,24)
square_xors=make_unpacked_configurations(square_largest_indices,24)

#a=triangle_xors.T*np.abs(triangle_largest_coeffs.flatten())
#triangle_xors_weighted=a.T

a=kagome_xors.T*np.abs(kagome_largest_coeffs.flatten())
kagome_xors_weighted=a.T
a=square_xors.T*np.abs(square_largest_coeffs.flatten())
square_xors_weighted=a.T

print('fourier transform finished')


features = [f"s{i}" for i in range(system.number_spins)]


square_head=square_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')
kagome_head=kagome_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')

for i in range(len(j2_array)):
	triangle_df=triangle_systems[i]
	s_name='triangle_j2='+str(j2_array[i])
	triangle_head=triangle_df.nlargest(N_COMPONENTS, ['amplitude'], keep='first')
	plot_umap(triangle_head[features].values, triangle_head['sign'].values,  triangle_head['amplitude'].values, s_name,j2_array[i])


plot_umap(square_head[features].values, square_head['sign'].values,  square_head['amplitude'].values, 'square', 1)
plot_umap(kagome_head[features].values, kagome_head['sign'].values, kagome_head['amplitude'].values, 'kagome', 1)



