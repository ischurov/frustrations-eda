from PIL import Image
import sys

N_COMPONENTS=sys.argv[1]

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
results_path="umap"

#N_COMPONENTS=30000


images_all=[results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=04.png', 
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=08.png', 
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=09.png', 
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=091.png',
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=092.png',
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=093.png',
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=094.png',
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=095.png',
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=1.png', 
	results_path+'/umap_fourier_'+str(N_COMPONENTS)+'_triangle_j2=125.png',]

combine_images(columns=10, space=5, images=images_all, name='umap_components_'+str(N_COMPONENTS)+'.png')

