import glob
import os
import sys
sys.path.append('..')

from predictor import Predictor

def run(myAnnFileName, buses):
    args = {'conf': 'config.json', 'weights':'weights_buses12.h5'}
    p = Predictor(args)
    files = glob.glob(buses + '/*.JPG')

    with open(myAnnFileName, 'w') as in_files:

        for eachfile in files:
            base_name = os.path.basename(eachfile)
            file_name = os.path.splitext(base_name)[0]

            annots = p.predict(eachfile)

            row = base_name + ":"

            for ant in annots:
                ant_formatted = [ant[0], ant[1], ant[2]-ant[0], ant[3]-ant[1], ant[4]]
                row = row + str(ant_formatted) + ','
            if row[-1] == ',':
                    row = row[:-1]

            in_files.write(row + '\n')
