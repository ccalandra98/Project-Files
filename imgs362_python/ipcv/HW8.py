if __name__ == '__main__':

         import cv2
         import ipcv
         import os.path
         import time
         import numpy

         home = os.path.expanduser('~')
         directory = home + os.path.sep + 'src/python/examples/data/bilateral'
         filename = home + os.path.sep + 'src/python/examples/data/panda_color.jpg'
         filename1 = home + os.path.sep + 'src/python/examples/data/panda.jpg'

         src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
         src = cv2.imread(filename1, cv2.IMREAD_UNCHANGED)




        #Create list of all image combinations to automatically run.
         d = (1,3,5,10)
         r = (10,30,50,100,300)
         combinations = numpy.zeros((2,len(d)*len(r)))
         i = 0
         j = 0
         count = 0
         for k in range(0,len(d)*len(r)):
            if i >4:
                i = 0
            if count >4:
                j = j+1
                count = 0
            combinations[0,k] = r[i]
            combinations[1,k] = d[j]
            i = i+1
            print(str)
            count = count+1

         #create list of names for images
         strings = (        'panda_d01_r010.png',
                            'panda_d01_r030.png',
                            'panda_d01_r050.png',
                            'panda_d01_r100.png',
                            'panda_d01_r300.png',
                            'panda_d03_r010.png',
                            'panda_d03_r030.png',
                            'panda_d03_r050.png',
                            'panda_d03_r100.png',
                            'panda_d03_r300.png',
                            'panda_d05_r010.png',
                            'panda_d05_r030.png',
                            'panda_d05_r050.png',
                            'panda_d05_r100.png',
                            'panda_d05_r300.png',
                    'panda_d10_r010.png',
                    'panda_d10_r030.png',
                    'panda_d10_r050.png',
                    'panda_d10_r100.png',
                    'panda_d10_r300.png')
         i = 0
         os.chdir(directory)
         for i in range(0,20):
            startTime = time.time()
            dst = ipcv.bilateral_filter(src, combinations[1,i], combinations[0,i], d=-1)
            print('Image Creation Time: = {0} [s]'.format(time.time() - startTime))

            savename = strings[i]
            cv2.imwrite(savename,dst)
            print('Successfully saved\n\n')
