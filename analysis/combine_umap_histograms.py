from PIL import Image
import sys

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


#combine_images(columns=2, space=20, images=['/vol/tcm11/kravchenko/correct_rdms_1000_weighted/kagome_corr.png', '/vol/tcm11/kravchenko/correct_rdms_1000_weighted/triangle_corr.png', '/vol/tcm11/kravchenko/correct_rdms_1000_weighted/kagome_corr_weighted.png', '/vol/tcm11/kravchenko/correct_rdms_1000_weighted/triangle_corr_weighted.png', '/vol/tcm11/kravchenko/triangle_vs_kagome.png', '/vol/tcm11/kravchenko/triangle_vs_kagome_weighted.png'])

'''
images_all=[]
for t in ['10', '100']:
	i1="natural_scale_impact_"+t+".png"
	images_all.append(i1)
	i2="artificial_scale_impact_"+t+".png"
	images_all.append(i2)
'''

results_path='results_polished_umap'

N_COMPONENTS=30000


lattice_array=[results_path+'/umap_lattice_30000_square.png', 
	results_path+'/umap_lattice_30000_triangle_j2=0.8.png',
	results_path+'/umap_lattice_30000_triangle_j2=1.png',
	results_path+'/umap_lattice_30000_kagome.png']


xor_array=[results_path+'/umap_fourier_30000_square.png', 
	results_path+'/umap_fourier_30000_triangle_j2=08.png',
	results_path+'/umap_fourier_30000_triangle_j2=1.png',
	results_path+'/umap_fourier_30000_kagome.png']

histograms_path='results_polished_histograms'


histograms_array=[histograms_path+'/square_coeffs_30000_first200.png',
	histograms_path+'/triangle_coeffs_30000_first200_j2=0.8.png',
	histograms_path+'/triangle_coeffs_30000_first200_j2=1.0.png',
	histograms_path+'/kagome_coeffs_30000_first200.png']

images_all=lattice_array+xor_array+histograms_array

combine_images(columns=4, space=0, images=images_all, name='umap_first_panel.png')

##############
#triangle versions!



results_path='umap'

N_COMPONENTS=30000


lattice_array=[results_path+'/umap_lattice_30000_triangle_j2=0.9.png',
	results_path+'/umap_lattice_30000_triangle_j2=0.92.png',
	results_path+'/umap_lattice_30000_triangle_j2=0.94.png',
	results_path+'/umap_lattice_30000_triangle_j2=0.95.png']


xor_array=[results_path+'/umap_fourier_30000_triangle_j2=09.png',
	results_path+'/umap_fourier_30000_triangle_j2=092.png',
	results_path+'/umap_fourier_30000_triangle_j2=094.png',
	results_path+'/umap_fourier_30000_triangle_j2=095.png']

histograms_path='histograms'


histograms_array=[histograms_path+'/triangle_coeffs_30000_first200_j2=0.9.png',
	histograms_path+'/triangle_coeffs_30000_first200_j2=0.92.png',
	histograms_path+'/triangle_coeffs_30000_first200_j2=0.94.png',
	histograms_path+'/triangle_coeffs_30000_first200_j2=0.95.png',]

images_all=lattice_array+xor_array+histograms_array

combine_images(columns=4, space=0, images=images_all, name='umap_second_panel.png')

'''


comp_n=[50, 100, 200, 300, 400, 500, 1000, 5000, 10000, 20000, 30000] 

images_all=[]

for c_n in comp_n:
	filen='umap_components_'+str(c_n)+'.png'
	images_all.append(filen)


combine_images(columns=1, space=5, images=images_all, name='umap_components_all.png')



images_all=[]

for c_n in comp_n:
	filen='umap_lattice_'+str(c_n)+'.png'
	images_all.append(filen)


combine_images(columns=1, space=5, images=images_all, name='umap_lattice_all.png')







first_n=200

all_images=[]
J2_array=[0.4, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 1.0, 1.25]

for i in range(len(J2_array)):
	J2_N=J2_array[i]
	fname=results_path+'/triangle_coeffs_'+str(N_COMPONENTS)+'_first'+str(first_n)+'_j2='+str(J2_N)+'.png'
	all_images.append(fname)


#'square_coeffs_'+str(N_COMPONENTS)+'.png',
combine_images(columns=10, space=3, images=all_images, name=results_path+"/all_triangle_coeffs_"+str(N_COMPONENTS)+'.png')

'''

