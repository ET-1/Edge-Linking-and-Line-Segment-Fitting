import cv2
from glob import glob
from utility import *

def main():
    # Image inputs path. Change it to your own input path.
    inpath = '/home/'

    # Output path. !!!!! All files will be cleaned before execution. !!!!!
    outpath = '/home/'
    if not os.path.exists(inpath):
        raise FileNotFoundError('No such input directory\n')

    if os.path.exists(outpath):
        clean_dir(outpath)
        create_dir(outpath)
    else:
        create_dir(outpath)

    
    file_all = glob(inpath + '/*jpg')       # Change the <jpg> extension as needed.
    os.chdir(outpath)
    # img_name = file_all[0:20]
    print('\nCalculation begins\n')
    for img_name in file_all:
        dirname, filename = os.path.split(img_name)
        fname, fename = os.path.splitext(filename)
        img = plt.imread(img_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # Convert the gray_img to gray
        #
        kernel_size = 7                                        # Kernel size for the Gaussian Blur
        blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 2)
        #
        low_threshold = 35                                     # Low threshold for Canny edge
        high_threshold = 60                                    # High threshold for Canny edge
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        A = edgelink(edges, 200)                               # 200 means nothing here. Don't mind it.
        A.get_edgelist()
        edgelist = A.edgelist
        etype = A.etype
        plt.imsave('%s.jpg' %fname, img)
        name = fname + '_canny'

        # Plot and save the Canny edge
        drawedgelist(edgelist, 'Ignore', 'rand', [img.shape[0], img.shape[1]], name)

        # Cluster all edges and filter edges whose lengths are less than 100 out
        nedgelist = cleanedgelist(edgelist.copy(), 100)
        name = fname + '_filter'
        # Plot and save the filtered edge
        drawedgelist(nedgelist, 'Ignore', 'rand', [img.shape[0], img.shape[1]], name)

        # Don't mind <tol> here. Usually it will not affect much things. But if you really want to figure it out
        # please read the paper <Visual Detection of Lintel-Occluded Doors from a Single Image>
        tol = 4
        sedgelist = seglist(nedgelist, tol)                    # Extract critical points of each edge class
        name = fname + '_segline'

        # Plot and save the critical points of each edge class
        drawedgelist(sedgelist, 'Ignore', 'rand', [img.shape[0], img.shape[1]], name)

        # Another filter. Filter edges whose lengths are less than 300 out.
        fedgelist = cleanedgelist(sedgelist.copy(), 300)
        name = fname + '_finaline'

        # Plot and save the filtered images
        drawedgelist(fedgelist, 'Ignore', 'rand', [img.shape[0], img.shape[1]], name)
        print('Finished edge link for %s' % fname)
    print('\nCalculation Done\n')

if __name__ == '__main__':
    main()