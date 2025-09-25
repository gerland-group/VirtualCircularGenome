#!/bin/env python3

class EnergyCalculator:

    def __init__(self, filepath_energy_parameters):

        self.filepath = filepath_energy_parameters
        self.read_block_energy_parameters()

    
    def read_block_energy_parameters(self):

        # open file
        f = open(self.filepath, 'r')
        filestring = f.read()
        f.close()

        # read lines from file
        lines = [el for el in filestring.split('\n') if el != '']

        # write block energy parameters
        self.block_energies = {}

        for line in lines:
            block, energy = line.split('=')
            block = block[1:-1]
            block = block.replace('X', ' ')
            l1,l2,l3,l4 = [el[1:-1] for el in block.split(',')]
            self.block_energies[(l1,l2,l3,l4)] = float(energy)


    def compute_energy_single_cmplx(self, cmplx_hr_nospecialchars):

        line1, line2 = cmplx_hr_nospecialchars.split('\n')
        
        if(len(line1) != len(line2)):
            raise ValueError('invalid complex string')

        energy = 0
        for i in range(len(line1)-1):
            
            # only interested in blocks that are not single-stranded
            if( (line1[i:i+2] != '  ') and (line2[i:i+2] != '  ')):

                l1 = line1[i+1]
                l2 = line2[i+1]
                l3 = line1[i]
                l4 = line2[i]

                block = (l1,l2,l3,l4)
                energy += self.block_energies[block]

        return energy


if __name__=='__main__':
    
    filepath = '../initial_configurations/energy_parameters.txt'
    ec = EnergyCalculator(filepath)
    cmplx_hr_nospecialchars_1 = 'AGAAAT\nCTTTAT'
    cmplx_hr_nospecialchars_2 = "AGAAAT \n CTTTAT"
    energy = ec.compute_energy_single_cmplx(cmplx_hr_nospecialchars_2)

