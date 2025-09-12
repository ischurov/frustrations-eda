
import matplotlib.pylab as plt
import pandas as pd
import pickle
import numpy as np
import os

import matplotlib

#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = "DejaVu Sans"

results_path="paper_final_xors"

list_dir=os.listdir(results_path)


kagome_dict = {"truncated": [], "flat": [], "nosign": []}
triangle_dict = {"truncated": [], "flat": [], "nosign": []}
square_dict = {"truncated": [], "flat": [], "nosign": []}

results_dict={"kagome": kagome_dict, "triangle": triangle_dict, "square": square_dict}


'''
kagome_truncated=[]
triangle_trucated=[]
square_truncated=[]

kagome_flat=[]
triangle_flat=[]
square_flat=[]

kagome_nosign=[]
triangle_nosign=[]
square_nosign=[]
'''
for fname in list_dir:  
	ftemp=fname.split(".")  #kagome_j2=1_flat_1725889966.7088835_modtest.pickle
	if (ftemp[-1]=='pickle'):
		if (fname[0:-7].split("_")[4]=='acc'):
			a=fname[0:-7].split("_")
			lattice=a[0]
			trunc=a[2]
			#test_set=[4]
			f=open(results_path+'/'+fname,'rb')
			acc=pickle.load(f)
			results_dict[lattice][trunc].append(acc)
			'''
			if lattice=='kagome': #FIXME this can be done more elegantly with dictionaries but also pfft
				if trunc=='truncated':
					kagome_truncated.append(a)
				if trunc=='flat':
					kagome_flat.append(a)
				if trunc=='nosign':
					kagome_nosign.append(a)
			if lattice=='triangle':
				if trunc=='truncated':
					triangle_truncated.append(a)
				if trunc=='flat':
					triangle_flat.append(a)
				if trunc=='nosign':
					triangle_nosign.append(a)
			if lattice=='square':
				if trunc=='truncated':
					square_truncated.append(a)
				if trunc=='flat':
					square_flat.append(a)
				if trunc=='nosign':
					square_nosign.append(a)
			'''

kagome_avg = {"truncated": [], "flat": [], "nosign": []}
triangle_avg = {"truncated": [], "flat": [], "nosign": []}
square_avg = {"truncated": [], "flat": [], "nosign": []}


kagome_std = {"truncated": [], "flat": [], "nosign": []}
triangle_std = {"truncated": [], "flat": [], "nosign": []}
square_std = {"truncated": [], "flat": [], "nosign": []}



results_avg={"kagome": kagome_avg, "triangle": triangle_avg, "square": square_avg}
results_std={"kagome": kagome_std, "triangle": triangle_std, "square": square_std}

#FIXME to get past a broken file
results_dict['square']['nosign'][5]=results_dict['square']['nosign'][4]

for key in results_dict:
	for kkey in results_dict[key]:
		results_avg[key][kkey]=np.mean(results_dict[key][kkey],0)	


for key in results_dict:
	print(key)
	for kkey in results_dict[key]:
		print(kkey)
		results_std[key][kkey]=np.std(results_dict[key][kkey],0)	


x=np.arange(10,510,10)

print('calculcated averages')

titledict={'kagome': 'kagome, $J_2=1$', 'triangle': 'triangle, $J_2=0.8$', 'square': 'square, $J_2=1$'}

for key in results_avg:
	plt.clf()
	print(key)
	'''
	plt.plot(x,results_avg[key]['truncated'],color='orange', linewidth=2, label='truncated')
	plt.fill_between(x, results_avg[key]['truncated']-results_std[key]['truncated'], results_avg[key]['truncated']+results_std[key]['truncated'])
	plt.plot(x,results_avg[key]['flat'],color='blue', linewidth=2, label='flat')
	plt.fill_between(x, results_avg[key]['flat']-results_std[key]['flat'], results_avg[key]['flat']+results_std[key]['flat'])
	plt.plot(x,results_avg[key]['nosign'],color='green', linewidth=2, label='nosign')
	plt.fill_between(x, results_avg[key]['nosign']-results_std[key]['nosign'], results_avg[key]['nosign']+results_std[key]['nosign'])
	#plt.plot(x,truncated_acc_orig,color='cyan', linewidth=2, label='train set: truncated, test set: original')
	#plt.plot(x,sign_acc,color='orange', linewidth=2, label='train set: non-weighted truncated')
	#plt.plot(x,sign_acc_orig,color='magenta', linewidth=2, label='train set: non-weighted truncated, test set: original')
	plt.xlabel('xors')
	plt.ylim(0.3,1.1)
	plt.ylabel('accuracy')
	plt.title(titledict[key]+', accuracy for 50 epochs')
	plt.legend(loc='lower right')
	plt.savefig(results_path+'/'+key+'.png')
	plt.savefig(results_path+'/'+key+'.eps')
	'''

	plt.clf()
	plt.plot(x,results_avg[key]['truncated'],color='#1e9a1b', linewidth=2, label='truncated')
	plt.fill_between(x, results_avg[key]['truncated']-results_std[key]['truncated'], results_avg[key]['truncated']+results_std[key]['truncated'],color='#bdd3bc', linewidth=0.5)
	plt.plot(x,results_avg[key]['flat'],color='#2471a3', linewidth=2, label='retained signs')
	plt.fill_between(x, results_avg[key]['flat']-results_std[key]['flat'], results_avg[key]['flat']+results_std[key]['flat'], color='#8cb5d0', linewidth=1.5)
	plt.plot(x,results_avg[key]['nosign'],color='orange', linewidth=2, label='homogenized')
	plt.fill_between(x, results_avg[key]['nosign']-results_std[key]['nosign'], results_avg[key]['nosign']+results_std[key]['nosign'],color='#fed5c0', linewidth=0.5)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlabel('number of terms', fontsize=20)
	plt.ylim(0.3,1)
	plt.ylabel('accuracy', fontsize=20)
	plt.title(titledict[key]+', 50 epochs', fontsize=20)
	lloc='lower right'
	if key=='kagome':
		lloc='upper right'
	plt.legend(loc=lloc, fontsize=18)
	ax = plt.subplot(111)
	ax.spines[['right', 'top']].set_visible(False)
	plt.tight_layout()
	plt.savefig(results_path+'/'+key+'_full.png')
	#plt.savefig(output_path+'/'+key+'_full.eps')



