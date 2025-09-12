
from pathlib import Path

import numpy as np

from fast_boolean_analysis import fourier_expand, keep_largest_n
from heisenberg_hamiltonians import HeisenbergJ1J2
from lattice_boolean_analysis import LBFFromSpinSystem, SignSignalKind
from spin_lattices import KagomeLattice, TriangleLattice, SquareLattice

import os
from os import path


from scipy.spatial import distance

from PIL import Image

import matplotlib
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = "DejaVu Sans"
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['figure.titlesize'] = 20


import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns 

import pickle

from utils import make_unpacked_configurations
from utils import get_abslargest_terms

N_COMPONENTS=30000
J2_N=0.8

results_path="results_polished_histograms"

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
	fig.savefig('umap/' + outname)
	plt.close()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i]+0.0001,(format(y[i], '.5f')))

def combine_images(columns, space, images, name):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(name)



triangle_lattice = TriangleLattice(6, 4)
kagome_lattice=KagomeLattice(2,4)
square_lattice=SquareLattice(6,4)

print('generated lattices')

system = HeisenbergJ1J2(triangle_lattice, J1=1, J2=J2_N, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
triangle_series = fourier_expand(lbf)


system = HeisenbergJ1J2(kagome_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
kagome_series = fourier_expand(lbf)


system = HeisenbergJ1J2(square_lattice, J1=1, J2=1, ground_state_cache_dir=Path("groundstates"))
lbf = LBFFromSpinSystem(system, eigenstate=0, kind=SignSignalKind())
square_series = fourier_expand(lbf)


print('fourier transform finished')

triangle_largest_indices, triangle_largest_coeffs = get_abslargest_terms(triangle_series.coeffs, N_COMPONENTS)
kagome_largest_indices, kagome_largest_coeffs = get_abslargest_terms(kagome_series.coeffs, N_COMPONENTS)
square_largest_indices, square_largest_coeffs = get_abslargest_terms(square_series.coeffs, N_COMPONENTS)




triangle_coef_sort=np.sort(np.abs(triangle_series.coeffs))
triangle_coef_truncated=triangle_coef_sort[-N_COMPONENTS:]

kagome_coef_sort=np.sort(np.abs(kagome_series.coeffs))
kagome_coef_truncated=kagome_coef_sort[-N_COMPONENTS:]

square_coef_sort=np.sort(np.abs(square_series.coeffs))
square_coef_truncated=square_coef_sort[-N_COMPONENTS:]



kagome_normalized=kagome_coef_truncated/np.sum(kagome_coef_truncated)
kagome_normalized=np.flip(kagome_normalized)

triangle_normalized=triangle_coef_truncated/np.sum(triangle_coef_truncated)
triangle_normalized=np.flip(triangle_normalized)

square_normalized=square_coef_truncated/np.sum(square_coef_truncated)
square_normalized=np.flip(square_normalized)



kagome_max=np.max(kagome_normalized)
square_max= np.max(square_normalized)
triangle_max=np.max(triangle_normalized)


'''
 # creating data on which bar chart will be plot
    x = ["Engineering", "Hotel Management", "MBA",
         "Mass Communication", "BBA", "BSc", "MSc"]
    y = [9330, 4050, 3030, 5500,
         8040, 4560, 6650]
     
    # making the bar chart on the data
    plt.bar(x, y)
     
    # calling the function to add value labels
    addlabels(x, y)
'''

ylim_max=0.0035#np.max(square_normalized)*1.1
ylim_mid=0.0035#np.max(triangle_normalized)*1.2
plt.clf()

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18,'axes.titlesize': 20, 'figure.titlesize' : 20})
#plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14,'axes.titlesize': 15, 'figure.titlesize' : 16})


x=list(range(0,N_COMPONENTS))
plt.clf()
plt.bar(x, triangle_normalized, align='center', alpha=0.5, ecolor='black', capsize=24)
addlabels([0], [triangle_max])
plt.title('triangular lattice, $J_2='+str(J2_N)+'$')
plt.ylim(0,ylim_max)
plt.tight_layout()
plt.savefig(results_path+'/triangle_coeffs_'+str(N_COMPONENTS)+'.png')


plt.clf()
plt.bar(x, kagome_normalized, align='center', alpha=0.5, ecolor='black', capsize=24)
plt.ylim(0,ylim_max)
addlabels([0], [kagome_max])
plt.title('kagome lattice, $J_2=1$')
plt.tight_layout()
plt.savefig(results_path+'/kagome_coeffs_'+str(N_COMPONENTS)+'.png')


plt.clf()
plt.bar(x, square_normalized, align='center', alpha=0.5, ecolor='black', capsize=24) #control bar width
plt.ylim(0,ylim_max)
addlabels([0], [square_max])
plt.title('square lattice, $J_2=1$')
plt.tight_layout()
plt.savefig(results_path+'/square_coeffs_'+str(N_COMPONENTS)+'.png')


first_n=200

x=list(range(0,first_n))
plt.clf()
plt.bar(x, triangle_normalized[0:first_n], align='center', alpha=0.5, ecolor='black', capsize=24)
plt.title('triangular lattice, $J_2='+str(J2_N)+'$')
plt.ylim(0,ylim_max)
addlabels([0], [triangle_max])
plt.tight_layout()
plt.savefig(results_path+'/triangle_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'.png')


plt.clf()
plt.bar(x, kagome_normalized[0:first_n], align='center', alpha=0.5, ecolor='black', capsize=24)
plt.ylim(0,ylim_max)
addlabels([0], [kagome_max])
plt.title('kagome lattice, $J_2=1$')
plt.tight_layout()
plt.savefig(results_path+'/kagome_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'.png')


plt.clf()
plt.bar(x, square_normalized[0:first_n], align='center', alpha=0.5, ecolor='black', capsize=24) #control bar width
plt.ylim(0,ylim_max)
addlabels([0], [square_max])
plt.title('square lattice, $J_2=1$')
plt.tight_layout()
plt.savefig(results_path+'/square_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'.png')




#'square_coeffs_'+str(N_COMPONENTS)+'.png',
combine_images(columns=3, space=3, images=[results_path+'/square_coeffs_'+str(N_COMPONENTS)+'.png', results_path+'/triangle_coeffs_'+str(N_COMPONENTS)+'.png', 
results_path+'/kagome_coeffs_'+str(N_COMPONENTS)+'.png'], name=results_path+"/all_coeffs_"+str(N_COMPONENTS)+'_j2='+str(J2_N)+'.png')

#'square_coeffs_'+str(N_COMPONENTS)+'.png',
combine_images(columns=3, space=3, images=[results_path+'/square_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'.png', results_path+'/triangle_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'.png', 
results_path+'/kagome_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'.png'], name=results_path+'/first'+str(first_n)+'_'+str(N_COMPONENTS)+'_j2='+str(J2_N)+'.png')



