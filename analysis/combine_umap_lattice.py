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


results_path="umap"

#N_COMPONENTS=30000


images_all=[results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.4.png', 
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.8.png', 
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.9.png', 
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.91.png',
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.92.png',
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.93.png',
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.94.png',
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=0.95.png',
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=1.png', 
	results_path+'/umap_lattice_'+str(N_COMPONENTS)+'_triangle_j2=1.25.png',]

combine_images(columns=10, space=5, images=images_all, name='umap_lattice_'+str(N_COMPONENTS)+'.png')

