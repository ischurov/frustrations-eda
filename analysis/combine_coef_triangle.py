
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
matplotlib.rcParams['text.usetex'] = True
#plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14,'axes.titlesize': 15, 'figure.titlesize' : 16})

import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns 

import pickle

from utils import make_unpacked_configurations
from utils import get_abslargest_terms

N_COMPONENTS=30000


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
	fig.savefig('/vol/tcm11/kravchenko/' + outname)
	plt.close()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i]+0.0001,y[i])

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


first_n=200

all_images=[]
J2_array=[0.4, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 1.0, 1.25]

for i in range(len(J2_array)):
	J2_N=J2_array[i]
	fname=results_path+'/triangle_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'_j2='+str(J2_N)+'.png'
	all_images.append(fname)


#'square_coeffs_'+str(N_COMPONENTS)+'.png',
combine_images(columns=10, space=3, images=all_images, name=results_path+"/all_triangle_coeffs_"+str(N_COMPONENTS)+'.png')



