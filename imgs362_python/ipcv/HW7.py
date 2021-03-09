if __name__ == '__main__':

         import cv2
         import fnmatch
         import numpy
         import os
         import os.path
         import ipcv
         import matplotlib.pyplot as plot


         home = os.path.expanduser('~')
         baseDirectory = home + os.path.sep + 'src/python/examples/data'
         baseDirectory += os.path.sep + 'character_recognition'

         documentFilename = baseDirectory + '/notAntiAliased/text.tif'
         documentFilename = baseDirectory + '/notAntiAliased/alphabet.tif'
        # documentFilename = baseDirectory + '/notAntiAliased/characters/39.tif'
         charactersDirectory = baseDirectory + '/notAntiAliased/characters'

         document = cv2.imread(documentFilename, cv2.IMREAD_UNCHANGED)

         characterImages = []
         characterCodes = []
         for root, dirnames, filenames in os.walk(charactersDirectory):
            for filename in sorted(filenames):
               currentCharacter = cv2.imread(root + os.path.sep + filename,
                                             cv2.IMREAD_UNCHANGED)
               characterImages.append(currentCharacter)
               code = int(os.path.splitext(os.path.basename(filename))[0])
               characterCodes.append(code)
         characterImages = numpy.asarray(characterImages)
         characterCodes = numpy.asarray(characterCodes)

         # Define the filter threshold
         threshold = 0.99

         text, histogram = ipcv.character_recognition(document,
                                                 characterImages,
                                                 characterCodes,
                                                 threshold,
                                                 filterType='matched')
         print(text)
         plot.show()
         # Display the results to the user

         text, histogram = ipcv.character_recognition(document,
                                                 characterImages,
                                                 characterCodes,
                                                 threshold,
                                                 filterType='spatial')

         # Display the results to the user
         print(text)
         plot.show()