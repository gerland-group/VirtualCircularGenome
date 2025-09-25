#!/bin/env python3

import numpy as np
import re
from copy import deepcopy

from helix import *
from strand import *
import helper_functions as hf


class Complex:
    
    """
    Object to hold a double stranded complex of hybridized oligomers.
    Takes the sequences of oligomers as well as the hybridized regions and returns
    complex in human-readable form. 

    Caveat: If the same oligomer appears multiple times in the complex, separate
    instances of Strand need to be created.
    
    """

    def __init__(self, helices=None, strand=None):

        if(helices != None and strand == None):
            self.helices = helices
            self.id2helix = {helix.id:helix for helix in self.helices}
            self.read_strands()
            self.id2strand = {strand.id:strand for strand in self.strands}     
            self.strandids_simp = ['_'.join(strand.id.split('_')[0:2]) \
                    for strand in self.strands] 
            
            # read which strands belong to which helices
            self.construct_map_strand_to_helices()

            # for each strand A determine the extremal helices and strands
            # i. e. helices hybridized to strand A that are furthest towards 5' and 3' end
            self.determine_extremal_helices_all_strands()

            # determine strands and helices that terminate the complex, 
            # i. e. left and right end
            self.determine_terminal_strands_and_helices()

            # construct humanreadable representation
            self.create_humanreadable_representation()
            self.create_humanreadable_representation_nospecialchars()

            # create the id
            self.create_id()

            # delete all variables that are not needed anymore
            del self.strand2helices
            del self.idstrand2idextremalhelices
            del self.idstrand2idextremalstrands
            if(hasattr(self, 'idstrand2idextremalstrands_hr')):
                del self.idstrand2idextremalstrands_hr
            del self.idstrand2type

            # determine which positions in the complex are occupied and empty
            self.determine_occupied_positions()
            self.determine_available_positions()

        
        elif(helices == None and strand != None):

            self.helices = []
            self.id2helix = {}

            self.strands = [strand]
            self.id2strand = {strand.id:strand}
            self.strandids_simp = ['_'.join(strand.id.split('_')[0:2])]

            self.top_3t5 = " "*(2*len(strand.seq_5t3)-1)
            self.bottom_5t3 = "-".join(re.findall('.', strand.seq_5t3))
            self.create_humanreadable_representation_nospecialchars()

            self.create_id()

            self.idstrand2pos = {strand.id:(0,-1)}

            self.l = strand.l

            self.determine_occupied_positions()
            self.determine_available_positions()
        
        elif(helices != None and strand != None):
            raise ValueError("helices and strand cannot be specified simultaneously")
    
        else:
            raise ValueError("neither helices nor strand specified")

    
    def __str__(self, ROTATED=False):
        
        if(not ROTATED):
            return self.top_3t5 + "\n" + self.bottom_5t3
        else:
            return self.bottom_5t3[::-1]+ "\n" +self.top_3t5[::-1]

    def __len__(self):

        return self.l

    def deepcopy(self):

        return deepcopy(self)
    
    
    def read_strands(self):

        self.strands = []
        for helix in self.helices:
            self.strands.append(helix.strand_bottom)
            self.strands.append(helix.strand_top)
        # self.strands = sorted(list(set(self.strands)), key = lambda strand: int(strand.id.split('_')[1]))
    
    
    def construct_map_strand_to_helices(self):

        self.strand2helices = {}

        for helix in self.helices:

            if(not (helix.strand_bottom in self.strand2helices)):
                self.strand2helices[helix.strand_bottom] = [helix]
            else:
                self.strand2helices[helix.strand_bottom].append(helix)
            if(not (helix.strand_top in self.strand2helices)):
                self.strand2helices[helix.strand_top] = [helix]
            else:
                self.strand2helices[helix.strand_top].append(helix)

    
    def determine_extremal_helices_single_strand(self, strand):

        if(strand in self.strand2helices):

            helices_bare = self.strand2helices[strand]
            helices = []
            for helix in helices_bare:
                if(helix.strand_top == strand):
                    helix.rotate()
                    helices.append(helix.rotated)
                elif(helix.strand_bottom == strand):
                    helices.append(helix)
                else:
                    raise ValueError("invalid helix entered in " \
                    "Complex:determine_extremal_hybridization_partners_single_strand")
            
            positions_of_hybridization_partners = [helix.start_top for helix in helices]

            index_5 = positions_of_hybridization_partners.index(\
                        min(positions_of_hybridization_partners))
            index_3 = positions_of_hybridization_partners.index(\
                        max(positions_of_hybridization_partners))
            
            helix_5 = helices[index_5]
            helix_3 = helices[index_3]

            strand_5 = helix_5.strand_top
            strand_3 = helix_3.strand_top

        else:
            raise ValueError("unexpected strand entered in " \
            "Complex:determine_extremal_hybridization_partners_single_strand")
        
        if( (helix_5 == helix_3) and (strand_5 == strand_3)):
            type = 'nc' # no continuation
        else:
            type = 'c' # continuation

        return helix_5.id, helix_3.id, strand_5.id, strand_3.id, type


    def determine_extremal_helices_all_strands(self, HR=False):

        self.idstrand2idextremalhelices = {}
        self.idstrand2idextremalstrands = {}
        self.idstrand2type = {}

        for strand in self.strands:
                      
            h5p, h3p, s5p, s3p, type = \
                self.determine_extremal_helices_single_strand(strand)
            
            self.idstrand2idextremalhelices[strand.id] = (h5p,h3p)
            self.idstrand2idextremalstrands[strand.id] = (s5p,s3p)
            self.idstrand2type[strand.id] = type

        if(HR): # create human-readable representation
            
            self.idstrand2idextremalstrands_hr = {}
            for key,value in self.idstrand2idextremalstrands.items():
                self.idstrand2idextremalstrands_hr[key.seq_5t3] = \
                    (value[0].seq_5t3, value[1].seq_5t3)


    def determine_terminal_strands_and_helices(self):

        terminal_strands = []
        terminal_helices = []

        for sp, hps in self.idstrand2idextremalhelices.items():
            if( (hps[0] == hps[1]) and (self.id2helix[hps[0]].type != 't') \
                and (self.id2helix[hps[0]].type != 'tinv') ):
                terminal_strands.append(sp)
                terminal_helices.append(hps[0])

        return terminal_strands, terminal_helices


    def determine_next_helix_and_strand(self, known_strand, known_helix):

        helices_ids = self.idstrand2idextremalhelices[known_strand.id]
        for helix_id in helices_ids:
            if(helix_id != known_helix.id):
                if(self.id2helix[helix_id].strand_top.id != known_strand.id):
                    return self.id2helix[helix_id].strand_top.id, helix_id
                elif(self.id2helix[helix_id].strand_bottom.id != known_strand):
                    return self.id2helix[helix_id].strand_bottom.id, helix_id
                else:
                    raise ValueError("no appropriate strand can be found")
        raise ValueError("no appropriate helix can be found")


    def create_humanreadable_representation(self):

        if(len(self.helices) == 1):
           
            # TODO: TEST THIS PART FOR case len(self.helices) == 1 more precisely
            self.helices[0].create_humanreadable_representation()
            self.bottom_5t3 = self.helices[0].bottom_5t3
            self.top_3t5 = self.helices[0].top_3t5
            
            # dictionary to store positions of all strands
            self.idstrand2pos = {}
            
            # determine first helix (has to have z-shape)
            h_old = self.helices[0]
            s_first = self.strands[0]

            if(h_old.type == 'z' and s_first == h_old.strand_top):
                s_old = h_old.strand_bottom
                v_pos = -1 # s_old is on bottom strand
                self.idstrand2pos = {s_first.id:(0,1), s_old.id:((-1)*h_old.start_top,-1)}
            
            elif(h_old.type == 'z' and s_first == h_old.strand_bottom):
                h_old.rotate()
                h_old = h_old.rotated
                s_old = h_old.strand_bottom
                v_pos = -1 # s_old is on bottom strand
                self.idstrand2pos = {s_first.id:(0,1), s_old.id:((-1)*h_old.start_top,-1)}

            elif(h_old.type == 's' and s_first == h_old.strand_bottom):
                s_old = h_old.strand_top
                v_pos = 1 # s_old is on top strand
                self.idstrand2pos = {s_first.id:(0,-1), s_old.id:(h_old.start_top,1)}

            elif(h_old.type == 's' and s_first == h_old.strand_top):
                h_old.rotate()
                h_old = h_old.rotated
                s_old = h_old.strand_top
                v_pos = 1
                self.idstrand2pos = {s_first.id:(0,-1), s_old.id:(h_old.start_top,1)}
            
            else:
                raise ValueError("unexpected initial strand")
            
            self.l = (max([len(self.top_3t5), len(self.bottom_5t3)])+1)//2
            if(len(self.top_3t5)<2*self.l-1):
                self.top_3t5 += " "*(2*self.l-1-len(self.top_3t5))
            elif(len(self.bottom_5t3)<2*self.l-1):
                self.bottom_5t3 += " "*(2*self.l-1-len(self.bottom_5t3))

            self.shift_and_sort_idstrand2pos()

            return None
                
        # dictionary to store positions of all strands
        self.idstrand2pos = {}

        # list of all helices that have not been used so far
        helices_unused = self.helices.copy()    

        # determine first helix (has to have z-shape)
        terminal_strands, terminal_helices = self.determine_terminal_strands_and_helices()
        s_first = self.id2strand[terminal_strands[0]]
        h_old = self.id2helix[terminal_helices[0]]

        if(h_old.type == 'z' and s_first == h_old.strand_top):
            s_old = h_old.strand_bottom
            v_pos = -1 # s_old is on bottom strand
            self.idstrand2pos = {s_first.id:(0,1), s_old.id:((-1)*h_old.start_top,-1)}
        
        elif(h_old.type == 'z' and s_first == h_old.strand_bottom):
            h_old.rotate()
            h_old = h_old.rotated
            s_old = h_old.strand_bottom
            v_pos = -1 # s_old is on bottom strand
            self.idstrand2pos = {s_first.id:(0,1), s_old.id:((-1)*h_old.start_top,-1)}

        elif(h_old.type == 's' and s_first == h_old.strand_bottom):
            s_old = h_old.strand_top
            v_pos = 1 # s_old is on top strand
            self.idstrand2pos = {s_first.id:(0,-1), s_old.id:(h_old.start_top,1)}

        elif(h_old.type == 's' and s_first == h_old.strand_top):
            h_old.rotate()
            h_old = h_old.rotated
            s_old = h_old.strand_top
            v_pos = 1
            self.idstrand2pos = {s_first.id:(0,-1), s_old.id:(h_old.start_top,1)}
        
        else:
            raise ValueError("unexpected initial strand")
        
        # create human-readable representation of first helix
        h_old.create_humanreadable_representation()
        bottom_5t3 = h_old.bottom_5t3
        top_3t5 = h_old.top_3t5

        # remove helix h1 from unused helices list
        helices_unused.remove(self.id2helix[h_old.id])
        
        while( (self.idstrand2type[s_old.id] == 'c') ):

            s_p, h_p = self.determine_next_helix_and_strand(s_old, h_old)
            h = self.id2helix[h_p]

            if(v_pos == -1):

                if(h.strand_bottom != s_old):
                    h.rotate()
                    h = h.rotated

                var = (2*(h.start_hyb - (h_old.end_hyb-h_old.start_hyb))+1)
                if(var < 0):
                    print("h_old: ")
                    print(h_old)
                    print("h: ")
                    print(h)
                    raise ValueError("invalid complex")

                self.idstrand2pos[h.strand_top.id] = ((len(top_3t5)+var)//2, 1)

                top_3t5 += " "*(2*(h.start_hyb - (h_old.end_hyb-h_old.start_hyb))+1) + \
                    '-'.join(re.findall('.', h.strand_top.seq_5t3[::-1]))

            else:
                
                if(h.strand_top != s_old):
                    h.rotate()
                    h = h.rotated
                
                var = (2*(-h.start_top - (h_old.end_hyb-h_old.start_hyb))+1)
                if(var < 0):
                    print("h_old: ")
                    print(h_old)
                    print("h: ")
                    print(h)
                    raise ValueError("invalid complex")

                self.idstrand2pos[h.strand_bottom.id] = ((len(bottom_5t3)+var)//2, -1)

                bottom_5t3 += " "*(2*(-h.start_top - (h_old.end_hyb-h_old.start_hyb))+1) + \
                    '-'.join(re.findall('.', h.strand_bottom.seq_5t3))

            # update
            s_old = self.id2strand[s_p]
            h_old = self.id2helix[h_p]
            helices_unused.remove(h_old)
            v_pos *= (-1)

        while( len(helices_unused) != 0):
    
            helix = helices_unused[0]

            if(helix.strand_bottom.id in self.idstrand2pos):

                strand = helix.strand_bottom

                if(self.idstrand2pos[strand.id][1] == -1): # strand on bottom

                    index = 2*(self.idstrand2pos[strand.id][0]+helix.start_top)
                    self.idstrand2pos[helix.strand_top.id] = (index//2,1)
                    seq_ins = helix.strand_top.seq_5t3[::-1]
                    str_ins = "-".join(re.findall('.', seq_ins))
                    top_3t5 = hf.replace_str(top_3t5, str_ins, index)

                elif(self.idstrand2pos[strand.id][1] == 1): # strand on top

                    index = 2*(self.idstrand2pos[strand.id][0] + helix.end_bottom - helix.end_hyb)
                    self.idstrand2pos[helix.strand_top.id] = (index//2,-1)
                    seq_ins = helix.strand_top.seq_5t3
                    str_ins = "-".join(re.findall('.', seq_ins))
                    bottom_5t3 = hf.replace_str(bottom_5t3, str_ins, index)

            elif(helix.strand_top.id in self.idstrand2pos):
                
                helix.rotate()
                helix = helix.rotated
                strand = helix.strand_bottom

                if(self.idstrand2pos[strand.id][1] == -1): # strand on bottom
                    
                    index = 2*(self.idstrand2pos[strand.id][0]+helix.start_top)
                    self.idstrand2pos[helix.strand_top.id] = (index//2,1)
                    seq_ins = helix.strand_top.seq_5t3[::-1]
                    str_ins = "-".join(re.findall('.', seq_ins))
                    top_3t5 = hf.replace_str(top_3t5, str_ins, index)

                elif(self.idstrand2pos[strand.id][1] == 1): # strand on top
                    
                    index = 2*(self.idstrand2pos[strand.id][0] + helix.end_bottom - helix.end_hyb)
                    self.idstrand2pos[helix.strand_top.id] = (index//2,-1)
                    seq_ins = helix.strand_top.seq_5t3
                    str_ins = "-".join(re.findall('.', seq_ins))
                    bottom_5t3 = hf.replace_str(bottom_5t3, str_ins, index)

            helices_unused.remove(self.id2helix[helix.id])

        self.top_3t5 = top_3t5
        self.bottom_5t3 = bottom_5t3
        self.l = (max([len(top_3t5), len(bottom_5t3)])+1)//2
        if(len(self.top_3t5)<2*self.l-1):
            self.top_3t5 += " "*(2*self.l-1-len(self.top_3t5))
        elif(len(self.bottom_5t3)<2*self.l-1):
            self.bottom_5t3 += " "*(2*self.l-1-len(self.bottom_5t3))
        if(len(self.top_3t5) != len(self.bottom_5t3)):
            raise ValueError('invalid human-readable representation')

        self.shift_and_sort_idstrand2pos()
    

    def create_humanreadable_representation_nospecialchars(self):
        if(not hasattr(self, 'top_3t5')):
            self.create_humanreadable_representation()
        
        self.top_3t5_nospecialchar = self.top_3t5[::2]
        self.bottom_5t3_nospecialchar = self.bottom_5t3[::2]


    def create_id(self):

        top_numb = int(hf.convert_spacelineAGCT_to_123456(self.top_3t5[::-1]), \
            base=alphabet+3)
        bottom_numb = int(hf.convert_spacelineAGCT_to_123456(self.bottom_5t3), \
            base=alphabet+3)

        self.id = "c%s_%s_%s" %(len(self.strands), min([top_numb,bottom_numb]), \
                                max([top_numb,bottom_numb]))


    def shift_and_sort_idstrand2pos(self):

        self.idstrand2pos = dict(sorted(self.idstrand2pos.items(), \
                                        key=lambda item: item[1][0]))
        
        i0 = list(self.idstrand2pos.values())[0][0]
        if(i0 < 0):
            for key, value in self.idstrand2pos.items():
                self.idstrand2pos[key] = (value[0] + (-i0), value[1])
    
    def construct_idstrand2pos_backward(self):
        
        self.idstrand2pos_backward = {}
        for idstrand, props in self.idstrand2pos.items():
            lstrand = self.id2strand[idstrand].l
            iold = props[0]
            inew = self.l-iold-lstrand
            self.idstrand2pos_backward[idstrand] = (inew,(-1)*props[1])


    def determine_occupied_positions(self):

        self.pos_occ_bottom = []
        self.pos_occ_top = []

        for id in self.idstrand2pos.keys():
            
            if(self.idstrand2pos[id][1]==1): # top strand
                start = self.idstrand2pos[id][0]
                end = self.idstrand2pos[id][0]+self.id2strand[id].l
                self.pos_occ_top.extend(range(start, end))
            
            elif(self.idstrand2pos[id][1]==-1): # bottom strand
                start = self.idstrand2pos[id][0]
                end = self.idstrand2pos[id][0]+self.id2strand[id].l
                self.pos_occ_bottom.extend(range(start, end))
    

    def determine_available_positions(self):

        self.pos_avail_bottom = []
        self.pos_avail_top = []
        for i in range(len(self)):
            if(not i in self.pos_occ_bottom):
                self.pos_avail_bottom.append(i)
            elif(not i in self.pos_occ_top):
                self.pos_avail_top.append(i)
            else:
                continue

        # group the lists into groups of consecutive available positions
        # self.pos_avail_bottom = []
        # for k, g in groupby(enumerate(pos_avail_bottom_ungrouped), lambda ix: ix[0]-ix[1]):
        #     self.pos_avail_bottom.append(list(map(itemgetter(1), g)))
        # 
        # self.pos_avail_top = []
        # for k, g in groupby(enumerate(pos_avail_top_ungrouped), lambda ix: ix[0]-ix[1]):
        #     self.pos_avail_top.append(list(map(itemgetter(1), g)))

    
    def add_strand(self, s, i, v_pos):

        if(s.id in self.id2strand):
            sin = deepcopy(s)
            sin.id = s.id + "_" + hf.create_random_hexstring()
        else:
            sin = s

        # add strand to data-structures pertaining to strands
        self.strands.append(sin)
        self.id2strand[sin.id] = sin
        self.idstrand2pos[sin.id] = (i, v_pos)
        self.strandids_simp.append('_'.join(sin.id.split('_')[0:2]))

        if(v_pos == 1): # strand to be added in top line
            
            # find the hybridization partner of the inserted strand
            ids = [key for key,value in self.idstrand2pos.items() if value[1] == -1]
            positions = np.asarray([value[0] for value in self.idstrand2pos.values() \
                if value[1] == -1])
            k = np.where( np.abs(positions-i) == np.min(np.abs(positions-i)) )[0][-1]

            id = ids[k] # id of hybridization partner
            j = positions[k] # position of hybridization partner

            # create helix for strand and hybridization partner
            h = Helix(self.id2strand[id], sin, i-j)

            # add helix to data-structures pertaining to helices
            self.helices.append(h)
            self.id2helix[h.id] = h

            # add strand to human-readable representation of complex
            
            # increase size of cmplx string to right or left
            if( 2*(i + sin.l) > len(self.top_3t5)):
                self.top_3t5 += (2*(sin.l+i)-len(self.top_3t5)-1)*" "
            if(i < 0):
                self.top_3t5 = 2*(-i)*" " + self.top_3t5
                self.bottom_5t3 = 2*(-i)*" " + self.bottom_5t3
            
            # insert into human-readalbe cmplx string
            if(i >= 0):
                self.top_3t5 = hf.replace_str(self.top_3t5, \
                    "-".join(re.findall('.', sin.seq_5t3[::-1])), 2*i)
            elif(i < 0):
                self.top_3t5 = hf.replace_str(self.top_3t5, \
                    "-".join(re.findall('.', sin.seq_5t3[::-1])), 0)

        else: # strand to be added in bottom line
            
            # find the hybridization partner of the inserted strand
            ids = [key for key,value in self.idstrand2pos.items() if value[1] == 1]
            positions = np.asarray([value[0] for value in self.idstrand2pos.values() \
                if value[1] == 1])
            k = np.where( np.abs(positions-i) == np.min(np.abs(positions-i)) )[0][-1]

            id = ids[k]
            j = positions[k]

            # create helix for strand and hybridization partner
            h = Helix(sin, self.id2strand[id], j-i)

            # add helix to data-structures pertaining to helices
            self.helices.append(h)
            self.id2helix[h.id] = h    

            # add strand to human-readable representation of complex
            
            # increase size of cmplx string to right or left
            if( 2*(i + sin.l) > len(self.bottom_5t3)):
                self.bottom_5t3 += (2*(sin.l+i)-len(self.bottom_5t3)-1)*" "
            if(i < 0):
                self.top_3t5 = 2*(-i)*" " + self.top_3t5
                self.bottom_5t3 = 2*(-i)*" " + self.bottom_5t3
            
            # insert into human-readalbe cmplx string
            if(i >= 0):
                self.bottom_5t3 = hf.replace_str(self.bottom_5t3, \
                    "-".join(re.findall('.', sin.seq_5t3)), 2*i)
            elif(i < 0):
                self.bottom_5t3 = hf.replace_str(self.bottom_5t3, \
                    "-".join(re.findall('.', sin.seq_5t3)), 0)
        
        self.l = (max([len(self.top_3t5), len(self.bottom_5t3)])+1)//2
        if(len(self.top_3t5)<2*self.l-1):
            self.top_3t5 += " "*(2*self.l-1-len(self.top_3t5))
        elif(len(self.bottom_5t3)<2*self.l-1):
            self.bottom_5t3 += " "*(2*self.l-1-len(self.bottom_5t3))
        if(len(self.top_3t5) != len(self.bottom_5t3)):
            raise ValueError('invalid human-readable representation')

        self.create_humanreadable_representation_nospecialchars()
        self.shift_and_sort_idstrand2pos()
        self.determine_occupied_positions()
        self.determine_available_positions()
        self.create_id()
    

    def check_if_contains_mismatches(self):

        if(not hasattr(self, 'top_3t5_nospecialchar')):
            self.create_humanreadable_representation_nospecialchars()
         
        self.is_mismatchfree = True

        for i in range(len(self.top_3t5_nospecialchar)):
            if( self.top_3t5_nospecialchar[i] == ' ' or self.bottom_5t3_nospecialchar[i] == ' '):
                continue

            if not hf.check_complementarity_single_letter(self.top_3t5_nospecialchar[i], self.bottom_5t3_nospecialchar[i]):
                self.is_mismatchfree = False
                break


    def compute_number_of_matches_and_mismatches(self):
        
        if(not hasattr(self, 'top_3t5_nospecialchar')):
            self.create_humanreadable_representation_nospecialchars()

        self.n_matches = 0
        self.n_mismatches = 0
        
        for i in range(len(self.top_3t5_nospecialchar)):
            if( self.top_3t5_nospecialchar[i] == ' ' or self.bottom_5t3_nospecialchar[i] == ' '):
                continue

            comp = hf.check_complementarity_single_letter(self.top_3t5_nospecialchar[i], self.bottom_5t3_nospecialchar[i])

            if(comp):
                self.n_matches += 1
            elif(not comp):
                self.n_mismatches += 1


    def compute_length_of_hybridization_sites(self):

        self.l_hyb = 0
        for helix in self.helices:
            self.l_hyb += helix.l_hyb


    def compute_mass(self):

        if(not hasattr(self, 'top_3t5_nospecialchar')):
            self.create_humanreadable_representation_nospecialchars()

        self.mass = 0
        self.mass += len(re.findall('[^\s]', self.top_3t5_nospecialchar))
        self.mass += len(re.findall('[^\s]', self.bottom_5t3_nospecialchar))


    def construct_producable_strands_via_templatedligation(self):

        self.prods = []
        self.educts = []
        self.temps = []

        pos_strand_top = {strandid:pos[0] for strandid,pos in self.idstrand2pos.items() \
                if pos[1] == 1}
        pos_strand_bottom = {strandid:pos[0] for strandid,pos in self.idstrand2pos.items() \
                if pos[1] == -1}

        for i in range(len(pos_strand_top)):
            for j in range(i+1, len(pos_strand_top)):
                sid1, p1 = list(pos_strand_top.items())[i]
                sid2, p2 = list(pos_strand_top.items())[j]
                if(p1 + self.id2strand[sid1].l == p2):
                    prod_seq = self.id2strand[sid2].seq_5t3 + self.id2strand[sid1].seq_5t3
                    self.educts.append((self.id2strand[sid2].seq_5t3,\
                                        self.id2strand[sid1].seq_5t3))
                    self.prods.append(prod_seq)
                    for sidb, pb in pos_strand_bottom.items():
                        if (pb < p1+self.id2strand[sid1].l) and (pb+self.id2strand[sidb].l >= p2):
                            self.temps.append(self.id2strand[sidb].seq_5t3)


        for i in range(len(pos_strand_bottom)):
            for j in range(i+1, len(pos_strand_bottom)):
                sid1, p1 = list(pos_strand_bottom.items())[i]
                sid2, p2 = list(pos_strand_bottom.items())[j]
                if(p1 + self.id2strand[sid1].l == p2):
                    prod_seq = self.id2strand[sid1].seq_5t3 + self.id2strand[sid2].seq_5t3
                    self.educts.append((self.id2strand[sid1].seq_5t3, \
                                        self.id2strand[sid2].seq_5t3))
                    self.prods.append(prod_seq)
                    for sidt, pt in pos_strand_top.items():
                        if (pt < p1+self.id2strand[sid1].l) and (pt+self.id2strand[sidt].l >= p2):
                            self.temps.append(self.id2strand[sidt].seq_5t3)
        
        if not len(self.prods) == len(self.temps):
            print(self)
            print(self.prods)
            print(self.temps)
        
        assert len(self.prods) == len(self.temps)


