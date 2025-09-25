#!/bin/env python3
import copy
from functools import partial
import gc
import multiprocessing as mp
import numba
import numpy as np
import sys
import tqdm

from genome import *
from strand import *
from helix import *
from complex import *
from energy_calculator import *


class CompoundContainer():

    def __init__(self, \
            filepath_strands, energy_calculator=None, cref_single=None, \
            read_copy_numbers=None, is_truncated=None, truncation_type = None, \
            cutoff = None):

        # path to file in which all possible strands are listed
        # (usually complexes.txt file from RNAReactor)
        self.filepath = filepath_strands
        
        # flag to determine if copy numbers are supposed to be read from file
        self.read_copy_numbers = read_copy_numbers

        # reference concentration for a oligomer with copy number N = 1
        self.cref_single = cref_single

        # direcotries to store all complexes
        self.cmplxs = {}
        self.cmplxid2index = {}

        # instance of energy calculator class
        self.ec = energy_calculator

        # decide if the list of included complexes is truncated
        # and what the cutoff is
        self.is_truncated = is_truncated

        # which criterion is used to truncated possible complexes
        # possible options: cutoff, mismatches
        self.truncation_type = truncation_type
        if(self.truncation_type == 'cutoff'):
            self.cutoff = cutoff
        
        # read single strands from file
        self.read_strands()


    def read_strands(self):

        f = open(self.filepath)
        filestring = f.read()
        f.close()

        paragraphs = filestring.split('\n\n')
        paragraphs[-1] = paragraphs[-1][0:-1]

        self.strands = []
        self.strandid2index = {}

        for ip, paragraph in enumerate(paragraphs):
            
            lines = paragraph.split('\n')

            N = int(lines[0].split("=")[1])

            seq = str(lines[2][2:-3])

            if(self.read_copy_numbers and self.cref_single != None):
                strand = Strand(seq, numb=N, conc=N*self.cref_single)
            elif(self.read_copy_numbers and self.cref_single == None):
                strand = Strand(seq, numb=N)
            elif(not self.read_copy_numbers):
                strand = Strand(seq)

            if(not strand.id in self.strandid2index):
                self.strands.append(strand)
                self.strandid2index[strand.id] = ip

            else:
                print("strand_id: ", strand.id)
                print("strand_seq: ", strand.seq_5t3)
                raise ValueError('same strand appears multiple times in %s' %self.filename)

        self.strand_lengths = list(set([strand.l for strand in self.strands]))

        # sort list of strands and dictionary
        # self.strandid2index = {}
        # self.strands = sorted(self.strands, key = lambda strand: int(strand.id.split('_')[1]))
        # for i, strand in enumerate(self.strands):
        #     self.strandid2index[strand.id] = i


    def group_strands_by_length(self):

        self.strands_by_length = {}
        for strand in self.strands:
            hf.add_or_include_in_dict(strand.l, self.strands_by_length, strand)


    def group_strands_by_length_and_terminalletters(self):

        self.strands_by_length_and_startletters = {}
        self.strands_by_length_and_endletters = {}

        for strand in self.strands:
            
            if(not strand.l in self.strands_by_length_and_startletters):
                self.strands_by_length_and_startletters[strand.l] = {}
            
            if(not strand.l in self.strands_by_length_and_endletters):
                self.strands_by_length_and_endletters[strand.l] = {}
            
            # start letters
            for i in range(strand.l):
                letters = strand.seq_5t3[0:i+1]
                hf.add_or_include_in_dict(letters, \
                    self.strands_by_length_and_startletters[strand.l], strand)
            
            # end letters
            for i in range(strand.l):
                letters = strand.seq_5t3[-1-i:]
                hf.add_or_include_in_dict(letters, \
                    self.strands_by_length_and_endletters[strand.l], strand)
            
    
    def compute_initial_strand_concentration(self):

        for strand in self.strands:
            strand.conc = strand.numb* self.cref_single

    
    def set_initial_strand_concentration_monomer_hexamer(self, cs_by_length):

        for strand in self.strands:
            strand.conc = cs_by_length[strand.l]


    def determine_possible_strand_locations_given_strandlength(self, cmplx, \
            strand_length):
        
        # determine all un-occupied positions on the top-part of the complex
        pos_avail_top_rel = copy.deepcopy(cmplx.pos_avail_top)

        if(0 in pos_avail_top_rel):
            pos_avail_top_rel.extend(range(-(strand_length-1), 0, 1))
        
        if(cmplx.l-1 in pos_avail_top_rel):
            pos_avail_top_rel.extend(range(cmplx.l, cmplx.l+strand_length-1, 1))

        pos_avail_top_rel = sorted(pos_avail_top_rel)
        
        # determine all un-occupied positions on the bottom-part of the complex
        pos_avail_bottom_rel = copy.deepcopy(cmplx.pos_avail_bottom)

        if(0 in pos_avail_bottom_rel ):
            pos_avail_bottom_rel.extend(range(-(strand_length-1), 0, 1))
        
        if(cmplx.l-1 in pos_avail_bottom_rel):
            pos_avail_bottom_rel.extend(range(cmplx.l, cmplx.l+strand_length-1, 1))

        pos_avail_bottom_rel = sorted(pos_avail_bottom_rel)
        
        # determine possible locations on top strand
        locations_top = []

        for i in range(0, len(pos_avail_top_rel)-strand_length+1):
        
            group = pos_avail_top_rel[i:i+strand_length]
            diff = group - np.arange(strand_length)

            if( np.all(diff == diff[0]) ):
                locations_top.append(group[0])
        
        # determine possible locations on bottom strand
        locations_bottom = []

        for i in range(0, len(pos_avail_bottom_rel)-strand_length+1):

            group = pos_avail_bottom_rel[i:i+strand_length]
            diff = group - np.arange(strand_length)

            if( np.all(diff == diff[0])):
                locations_bottom.append(group[0])
        
        del pos_avail_bottom_rel
        del pos_avail_top_rel

        return locations_top, locations_bottom
    

    def determine_possible_strand_locations_given_strand(self, cmplx, strand):

        strand_length = strand.l

        return self.determine_possible_strand_locations_given_strandlength(\
            cmplx, strand_length)


    def identify_complementary_strands_single_strandlength(self, cmplx, \
        strand_length):

        locations_top, locations_bottom = \
            self.determine_possible_strand_locations_given_strandlength(\
            cmplx, strand_length)

        cmplxs_loc = []

        # strand added to bottom
        for i in locations_bottom:

            # strand does neither protrude to the left nor the right
            if( (i >= 0) and (i+strand_length <= cmplx.l) ):
                seq_top_3t5 = cmplx.top_3t5[::2][i:i+strand_length]
                seq_bottom_5t3 = hf.construct_complementary_strand_rev2fwd(seq_top_3t5)

                if(seq_bottom_5t3 in self.strands_by_length_and_startletters[strand_length]):
                    strands = self.strands_by_length_and_startletters[strand_length][seq_bottom_5t3]
                    for strand in strands:
                        cmplx_loc = deepcopy(cmplx)
                        cmplx_loc.add_strand(strand, i, -1)
                        cmplxs_loc.append(cmplx_loc)
            
            # strand does protrude to the left
            elif(i < 0):
                seq_top_3t5 =  cmplx.top_3t5[::2][0:i+strand_length]
                seq_bottom_5t3 = hf.construct_complementary_strand_rev2fwd(seq_top_3t5)

                if(seq_bottom_5t3 in self.strands_by_length_and_endletters[strand_length]):
                    strands = self.strands_by_length_and_endletters[strand_length][seq_bottom_5t3]
                    for strand in strands:
                        cmplx_loc = deepcopy(cmplx)
                        cmplx_loc.add_strand(strand, i, -1)
                        cmplxs_loc.append(cmplx_loc)
            
            # strand does protrude to the right
            elif(i+strand_length > cmplx.l):
                seq_top_3t5 = cmplx.top_3t5[::2][i:cmplx.l]
                seq_bottom_5t3 = hf.construct_complementary_strand_rev2fwd(seq_top_3t5)

                if(seq_bottom_5t3 in self.strands_by_length_and_startletters[strand_length]):
                    strands = self.strands_by_length_and_startletters[strand_length][seq_bottom_5t3]
                    for strand in strands:
                        cmplx_loc = deepcopy(cmplx)
                        cmplx_loc.add_strand(strand, i, -1)
                        cmplxs_loc.append(cmplx_loc)
        
        # strands added to top
        for i in locations_top:

            # strand does neither protrude to the left nor the right
            if( (i >= 0) and (i+strand_length <= cmplx.l) ):
                seq_bottom_5t3 = cmplx.bottom_5t3[::2][i:i+strand_length]
                seq_top_5t3 = hf.construct_complementary_strand_fwd2fwd(seq_bottom_5t3)

                if(seq_top_5t3 in self.strands_by_length_and_startletters[strand_length]):
                    strands = self.strands_by_length_and_startletters[strand_length][seq_top_5t3]
                    for strand in strands:
                        cmplx_loc = deepcopy(cmplx)
                        cmplx_loc.add_strand(strand, i, 1)
                        cmplxs_loc.append(cmplx_loc)
            
            # strand protrudes to the left
            elif(i <= 0):
                seq_bottom_5t3 = cmplx.bottom_5t3[::2][0:i+strand_length]
                seq_top_5t3 = hf.construct_complementary_strand_fwd2fwd(seq_bottom_5t3)

                if(seq_top_5t3 in self.strands_by_length_and_startletters[strand_length]):
                    strands = self.strands_by_length_and_startletters[strand_length][seq_top_5t3]
                    for strand in strands:
                        cmplx_loc = deepcopy(cmplx)
                        cmplx_loc.add_strand(strand, i, 1)
                        cmplxs_loc.append(cmplx_loc)
            
            # strand protrudes to the right
            elif(i+strand_length > cmplx.l):
                seq_bottom_5t3 = cmplx.bottom_5t3[::2][i:cmplx.l]
                seq_top_5t3 = hf.construct_complementary_strand_fwd2fwd(seq_bottom_5t3)
                
                if(seq_top_5t3 in self.strands_by_length_and_endletters[strand_length]):
                    strands = self.strands_by_length_and_endletters[strand_length][seq_top_5t3]
                    for strand in strands:
                        cmplx_loc = deepcopy(cmplx)
                        cmplx_loc.add_strand(strand, i, 1)
                        cmplxs_loc.append(cmplx_loc)

        return cmplxs_loc


    def identify_complementary_strands_single_multiple_strandlengths(self, \
            cmplx, strand_lengths):
        
        cmplxs_loc = []
        
        for strand_length in strand_lengths:
            cmplxs_loc_inter = self.identify_complementary_strands_single_strandlength(\
                cmplx, strand_length)
            cmplxs_loc.extend(cmplxs_loc_inter)

        return cmplxs_loc
    

    def compute_relevance_factor(self, cmplx):

        cmplx_hr_nospecialchars = cmplx.top_3t5_nospecialchar+'\n'+cmplx.bottom_5t3_nospecialchar
        energy = self.ec.compute_energy_single_cmplx(cmplx_hr_nospecialchars)
        Kd = np.exp(energy)
        cmplx.Kd = Kd

        concprod = 1
        for strand in cmplx.strands:
            concprod *= strand.conc
        
        return concprod/Kd


    def list_all_duplexes(self):

        self.cmplxs[2] = []
        self.cmplxid2index[2] = {}

        counter = 0
        for is1 in tqdm.tqdm(range(len(self.strands)), total=len(self.strands)):
            strand1 = self.strands[is1]
            cmplx1 = Complex(strand=strand1)
            
            for is2 in range(is1,len(self.strands)):
                # this choice of index for is2 should ensure that no complexes are double-counted
                strand2 = self.strands[is2]
            
                if(not ((strand1.l == 1) and (strand2.l == 1)) ): # monomers do not hybridize

                    loc_top, loc_bottom = \
                        self.determine_possible_strand_locations_given_strand(\
                        cmplx1, strand2)

                    identical = False                    
                    if( strand1.id == strand2.id ):
                            identical = True

                    for loc in loc_top:
                        cmplx_loc = copy.deepcopy(cmplx1)
                        strand_loc = copy.deepcopy(strand2)

                        if identical:
                            strand_loc.id = strand2.id + "_" + hf.create_random_hexstring()
                        
                        cmplx_loc.add_strand(strand_loc, loc, 1)

                        if (self.is_truncated and self.truncation_type == 'cutoff'):
                           rf = self.compute_relevance_factor(cmplx_loc)
                           if(rf >= self.cutoff):
                                self.cmplxs[2].append(cmplx_loc)
                                self.cmplxid2index[2][cmplx_loc.id] = counter
                                counter += 1

                        elif (self.is_truncated and self.truncation_type == 'mismatch'):
                            cmplx_loc.check_if_contains_mismatches()
                            if cmplx_loc.is_mismatchfree:
                                self.cmplxs[2].append(cmplx_loc)
                                self.cmplxid2index[2][cmplx_loc.id] = counter
                                counter += 1

                        elif (self.is_truncated and self.truncation_type == 'single_mismatch'):
                            cmplx_loc.compute_number_of_matches_and_mismatches()
                            if cmplx_loc.n_mismatches <= 1:
                                self.cmplxs[2].append(cmplx_loc)
                                self.cmplxid2index[2][cmplx_loc.id] = counter
                                counter += 1

                        else:
                            self.cmplxs[2].append(cmplx_loc)
                            self.cmplxid2index[2][cmplx_loc.id] = counter
                            counter += 1

                    if(len(loc_bottom) != 0):
                        raise ValueError("unexpected number of available positions\
                                         on bottom of double stranded complex")
        
    
    def list_all_nplus1_plexes_onestrand(self, strand, n):

        # TODO: might be changed, because complex object is automatically 
        # able to creat a copy of the strand if already included strand is
        # added

        cmplxs_loc = []
        
        for j, cmplx in enumerate(self.cmplxs[n]):
            
            loc_top, loc_bottom = self.determine_possible_strand_locations_given_strand(\
                cmplx, strand)
            
            # -> not necessary anymore
            identical = False
            if( strand.id in [strand.id for strand in cmplx.strands]):
                identical = True
            
            for loc in loc_top:
                cmplx_loc = copy.deepcopy(cmplx)
                
                # -> not necessary anymore
                strand_loc = copy.deepcopy(strand)
                if identical:
                    strand_loc.id = strand.id + "_" + hf.create_random_hexstring()
                
                cmplx_loc.add_strand(strand_loc, loc, 1)
                if (self.is_truncated and self.truncation_type == 'cutoff'):
                    rf = self.compute_relevance_factor(cmplx_loc)
                    if(rf >= self.cutoff):
                        cmplxs_loc.append(cmplx_loc)

                elif (self.is_truncated and self.truncation_type == 'mismatch'):
                    cmplx_loc.check_if_contains_mismatches()
                    if cmplx_loc.is_mismatchfree:
                        cmplxs_loc.append(cmplx_loc)
                
                elif (self.is_truncated and self.truncation_type == 'single_mismatch'):
                    cmplx_loc.compute_number_of_matches_and_mismatches()
                    if cmplx_loc.n_mismatches <= 1:
                        cmplxs_loc.append(cmplx_loc)

                else:
                    cmplxs_loc.append(cmplx_loc)
                
                del cmplx_loc
                del strand_loc
            
            for loc in loc_bottom:
                cmplx_loc = copy.deepcopy(cmplx)
                
                # -> not necessary anymore
                strand_loc = copy.deepcopy(strand)
                if identical:
                    strand_loc.id = strand.id + "_" + hf.create_random_hexstring()
                
                cmplx_loc.add_strand(strand_loc, loc, -1)
                if (self.is_truncated and self.truncation_type == 'cutoff'):
                    rf = self.compute_relevance_factor(cmplx_loc)
                    if(rf >= self.cutoff):
                        cmplxs_loc.append(cmplx_loc)
                
                elif (self.is_truncated and self.truncation_type == 'mismatch'):
                    cmplx_loc.check_if_contains_mismatches()
                    if cmplx_loc.is_mismatchfree:
                        cmplxs_loc.append(cmplx_loc)

                else:
                    cmplxs_loc.append(cmplx_loc)
                
                del cmplx_loc
                del strand_loc

        gc.collect()
        
        return cmplxs_loc
    

    def list_all_nplus1_plexes(self, n):

        """
        # set up multiple threads
        p = mp.Pool(1)
        
        # loop over all strands that can hybridize to existing complexes
        list_nplus1_plexes_interim = partial(self.list_all_nplus1_plexes_onestrand, n=n)
        nplus1_plexes_groups = p.map(list_nplus1_plexes_interim, self.strands)

        # concatenate all complexes into one list, remove redundancies
        self.cmplxs[n+1] = []
        self.cmplxid2index[n+1] = {}
        counter = 0

        # write constructed cmplxs to final cmplx dictionary
        for nplus1_plexes_group in tqdm.tqdm(nplus1_plexes_groups, total=len(nplus1_plexes_groups)):
            for cmplx in nplus1_plexes_group:
                if(not cmplx.id in self.cmplxid2index[n+1]):
                    self.cmplxid2index[n+1][cmplx.id] = counter
                    self.cmplxs[n+1].append(cmplx)
                    counter += 1

        del nplus1_plexes_groups
        gc.collect()
        """

        # concatenate all complexes into one list, remove redundancies
        self.cmplxs[n+1] = []
        self.cmplxid2index[n+1] = {}
        counter = 0

        for strand in tqdm.tqdm(self.strands, total=len(self.strands)):
            cmplxs_loc = self.list_all_nplus1_plexes_onestrand(strand, n)
            for cmplx_loc in cmplxs_loc:
                if(not cmplx_loc.id in self.cmplxid2index[n+1]):
                    self.cmplxid2index[n+1][cmplx_loc.id] = counter
                    self.cmplxs[n+1].append(cmplx_loc)
                    counter += 1
        
    
    def list_all_nplus1_plexes_no_mismatches_multithreading(self, n, imin=None, imax=None):
        
        # set up multiple threads
        #p = mp.Pool(mp.cpu_count()-4)
        p = mp.Pool(1)

        identify_complementary_strands_interim = partial(\
            self.identify_complementary_strands_single_multiple_strandlengths, \
            strand_lengths = self.strand_lengths)
        
        # nplus1_plexes_groups = p.map(identify_complementary_strands_interim, \
        #                             self.cmplxs[n])

        nplus1_plexes_groups = []
        if( (imin != None) and (imax != None)):

            for i in range(imin,imax,1):
                print("%s / %s" %(i, len(self.cmplxs[n])))
                cmplx = self.cmplxs[n][i]
                nplus1_plexes_groups.append(identify_complementary_strands_interim(cmplx))
                    
        else:
            for i, cmplx in enumerate(self.cmplxs[n]):
                print("%s / %s" %(i, len(self.cmplxs[n])))
                nplus1_plexes_groups.append(identify_complementary_strands_interim(cmplx))

        # concatenate all complexes into one list, remove redundancies
        self.cmplxs[n+1] = []
        self.cmplxid2index[n+1] = {}
        counter = 0

        # write constructed complexes to final cmplx dictionary
        for nplus1_plexes_group in nplus1_plexes_groups:
            for cmplx in nplus1_plexes_group:
                if(not cmplx.id in self.cmplxid2index[n+1]):
                    self.cmplxid2index[n+1][cmplx.id] = counter
                    self.cmplxs[n+1].append(cmplx)
                    counter += 1
        
        del nplus1_plexes_groups
        gc.collect()

    
    def list_all_nplus1_plexes_no_mismatches(self, n, VERBOSE=False):

        self.cmplxs[n+1] = []
        self.cmplxid2index[n+1] = {}
        counter = 0

        for i, cmplx in tqdm.tqdm(enumerate(self.cmplxs[n]), total=len(self.cmplxs[n])):
            cmplxs_loc = self.identify_complementary_strands_single_multiple_strandlengths(\
                    cmplx, self.strand_lengths)

            for cmplx_loc in cmplxs_loc:
                if(not cmplx_loc.id in self.cmplxid2index[n+1]):
                    self.cmplxid2index[n+1][cmplx_loc.id] = counter
                    self.cmplxs[n+1].append(cmplx_loc)
                    counter += 1


    def list_productive_complexes(self, genome=None):

        self.prods = {}

        for n in self.cmplxs.keys():
            for cmplx in self.cmplxs[n]:
                cmplx.construct_producable_strands_via_templatedligation()
                if(len(cmplx.prods) != 0):
                    for el in cmplx.prods:
                        if(genome == None):
                            hf.add_or_include_in_dict(len(el), self.prods, (el, cmplx.id))
                        elif(type(genome) == Genome):
                            incl = el in genome.included_words[len(el)]
                            hf.add_or_include_in_dict((len(el),incl), self.prods, \
                                                      (el, cmplx.id))
                        else:
                            raise ValueError('incorrect genome in list_productive_complexes')


    def list_productive_complexes_oligomerization_primerextension_ligation(self, \
            genome, l_unique):

        self.property2category = {(0,True):'oligomerization_correct', \
                                  (0,False):'oligomerization_false', \
                                  (1,True):'primer_extension_correct',\
                                  (1,False):'primer_extension_false',\
                                  (2,True):'ligation_correct',\
                                  (2,False):'ligation_false'}

        self.prods_opl = {}
        for n in self.cmplxs.keys():
            for cmplx in self.cmplxs[n]:
                cmplx.construct_producable_strands_via_templatedligation()
                if(len(cmplx.prods) != 0):
                    for i in range(len(cmplx.prods)):
                            
                            # check the length of the reacting starnds
                            educts = cmplx.educts[i]
                            if (len(educts[0]) < l_unique) and (len(educts[1]) < l_unique):
                                properties = [0,None]
                            elif (len(educts[0]) >= l_unique) and (len(educts[1]) < l_unique):
                                properties = [1,None]
                            elif (len(educts[0]) < l_unique) and (len(educts[1]) >= l_unique):
                                properties = [1,None]
                            elif (len(educts[0]) >= l_unique) and (len(educts[1]) >= l_unique):
                                properties = [2,None]
                            
                            # check if the produced sequence is 
                            prod = cmplx.prods[i]
                            incl = prod in genome.included_words[len(prod)]
                            properties[1] = incl

                            # identify the type of reaction 
                            properties = tuple(properties)
                            category = self.property2category[properties]

                            hf.add_or_include_in_dict(category, self.prods_opl, \
                                                      (prod, cmplx.id))
    
    
    def list_productive_complexes_cvf_lvs(self, l_unique, genome):

        self.reaction_types_cvflvs = ['ff_f_f_c', 'ff_f_v_c', 'ff_f_v_f', 'ff_v_f_c', \
                               'ff_v_v_c', 'ff_v_v_f', 'fv_f_v_c', 'fv_f_v_f', \
                               'fv_v_v_c', 'fv_v_v_f', 'vv_f_v_c', 'vv_f_v_f', \
                               'vv_v_v_c', 'vv_v_v_f']
        
        self.reaction_types_cvflvs_to_cmplx = {rt:[] for rt in self.reaction_types_cvflvs}

        for n in self.cmplxs.keys():
            for cmplx in self.cmplxs[n]:
                cmplx.construct_producable_strands_via_templatedligation()
                if(len(cmplx.prods) != 0):
                    for i in range(len(cmplx.prods)):

                        reaction_type = ""

                        # check the length of the reacting strands
                        educts = cmplx.educts[i]
                        if len(educts[0]) < l_unique and len(educts[1]) < l_unique:
                            reaction_type += "ff_"
                        elif len(educts[0]) >= l_unique and len(educts[1]) < l_unique:
                            reaction_type += "fv_"
                        elif len(educts[0]) < l_unique and len(educts[1]) >= l_unique:
                            reaction_type += "fv_"
                        else:
                            reaction_type += "vv_"

                        temp = cmplx.temps[i]
                        if len(temp) < l_unique:
                            reaction_type += "f_"
                        else:
                            reaction_type += "v_"
                        
                        prod = cmplx.prods[i]
                        if len(prod) < l_unique:
                            reaction_type += "f_"
                        else:
                            reaction_type += "v_"
                        
                        gencomp = prod in genome.included_words[len(prod)]
                        if gencomp:
                            reaction_type += 'c'
                        else:
                            reaction_type += 'f'

                        hf.add_or_include_in_dict(reaction_type, self.reaction_types_cvflvs_to_cmplx, cmplx.id)


    def list_complexes_by_type(self):

        first_cmplx = self.cmplxs[next(iter(self.cmplxs))][0]

        if(not hasattr(first_cmplx, 'energy')):
            self.compute_dissociation_constants_all_cmplxs()

        if(not hasattr(first_cmplx, 'l_hyb')):
            self.compute_length_of_hybridization_sites_all_cmplxs()
        
        cmplxs_by_type_inter = {}

        for strand_id, strand_index in self.strandid2index.items():
            hf.add_or_increase_in_dict((1,0,0), cmplxs_by_type_inter, 1)

        for nstrands in self.cmplxid2index.keys():
            for cmplx_id, cmplx_index in self.cmplxid2index[nstrands].items():
                lhyb = self.cmplxs[nstrands][cmplx_index].l_hyb
                dG = self.cmplxs[nstrands][cmplx_index].energy
                hf.add_or_increase_in_dict((nstrands,lhyb,dG), cmplxs_by_type_inter, 1)
        
        keys_sorted = sorted(list(cmplxs_by_type_inter.keys()))
        self.cmplxs_by_type = {key:cmplxs_by_type_inter[key] for key in keys_sorted}



    def compute_dissociation_constants_all_cmplxs(self):

        for nc in self.cmplxs.keys():
            for cmplx in self.cmplxs[nc]:
                cmplx_hr_nospecialchars = cmplx.top_3t5_nospecialchar+'\n'+cmplx.bottom_5t3_nospecialchar
                energy = self.ec.compute_energy_single_cmplx(cmplx_hr_nospecialchars)
                cmplx.energy = energy
                cmplx.Kd = np.exp(cmplx.energy)

    def compute_masses_all_cmplxs(self):

        for nc in self.cmplxs.keys():
            for cmplx in self.cmplxs[nc]:
                cmplx.compute_mass()


    def compute_length_of_hybridization_sites_all_cmplxs(self):

        for nc in self.cmplxs.keys():
            for cmplx in self.cmplxs[nc]:
                cmplx.compute_length_of_hybridization_sites()


    def compute_number_of_matches_and_mismatches_all_cmplxs(self):

        for nc in self.cmplxs.keys():
            for cmplx in self.cmplxs[nc]:
                cmplx.compute_number_of_matches_and_mismatches()
 
    
    def construct_cmplx_properties_flat(self):

        self.strand_indices_flat = []
        self.Kds_flat = []
        self.masses_flat = []
        self.lhybs_flat = []
        self.nmatches_flat = []
        self.nmismatches_flat = []
        
        for n in self.cmplxs:
            for cmplx in self.cmplxs[n]:
                strand_indices_loc = np.asarray([self.strandid2index[id] \
                        for id in cmplx.strandids_simp], dtype=int)
                self.strand_indices_flat.append(strand_indices_loc)
                self.Kds_flat.append(cmplx.Kd)
                self.masses_flat.append(cmplx.mass)
                self.lhybs_flat.append(cmplx.l_hyb)
                self.nmatches_flat.append(cmplx.n_matches)
                self.nmismatches_flat.append(cmplx.n_mismatches)

        self.strand_indices_flat = numba.typed.List(self.strand_indices_flat)
        self.Kds_flat = np.asarray(self.Kds_flat)
        self.masses_flat = np.asarray(self.masses_flat)
        self.lhybs_flat = np.asarray(self.lhybs_flat)
        self.nmatches_flat = np.asarray(self.nmatches_flat)
        self.nmismatches_flat = np.asarray(self.nmismatches_flat)
   

    def construct_strand_indices_sorted(self):

        self.strand_indices_sorted = sorted(np.arange(len(self.strands)), \
                key = lambda k: int(self.strands[k].id.split('_')[1]))

    
    def construct_cmplx_indices_flat_sorted(self):

        nstrands_flat = [len(el) for el in self.strand_indices_flat]

        self.props = []
        for cmplx_id, i in self.cmplxid2indexflat.items():
            self.props.append( (self.masses_flat[i], nstrands_flat[i], \
                self.nmismatches_flat[i]-self.nmatches_flat[i], \
                int(cmplx_id.split('_')[1]), int(cmplx_id.split('_')[2])) )

        self.cmplx_indices_flat_sorted = sorted(range(len(self.props)), \
                key = lambda k: self.props[k])


    def construct_sorted_strandids_and_sorted_cmplxids(self, make_private=False):

        if(not hasattr(self, 'strand_indices_sorted')):
            self.construct_strand_indices_sorted()

        strandids_sorted = [self.strands[i].id for i in self.strand_indices_sorted]

        if(not hasattr(self, 'cmplx_indices_flat_sorted')):
            self.construct_cmplx_indices_flat_sorted()

        cmplxindexflat2id = {value:key for key,value in self.cmplxid2indexflat.items()}
        cmplxids_sorted = [cmplxindexflat2id[index] for index \
                in self.cmplx_indices_flat_sorted]

        if make_private:
            self.__strandids_sorted = strandids_sorted
            self.__cmplxids_sorted = cmplxids_sorted

        else:
            self.strandids_sorted = strandids_sorted
            self.cmplxids_sorted = cmplxids_sorted


    def save_sorted_strandids_and_sorted_cmplxids(self, filepath):

        filestring = ""
        for strandid in self.strandids_sorted:
            filestring += strandid + "\n"

        for cmplxid in self.cmplxids_sorted:
            filestring += cmplxid + "\n"

        filestring = filestring[0:-1]

        # write to file
        f = open(filepath, 'w')
        f.write(filestring)
        f.close()


    def check_if_order_of_sorted_cmplxids_agrees_with_expected_order(self):

        self.construct_sorted_strandids_and_sorted_cmplxids()
        
        has_correct_strands = False
        has_correct_cmplxs = False

        if(self.strandids_sorted == self._CompoundContainer__strandids_sorted):
            has_correct_strands = True

        if(self.cmplxids_sorted == self._CompoundContainer__cmplxids_sorted):
            has_correct_cmplxs = True
     
        return has_correct_strands and has_correct_cmplxs

    
    def check_if_order_of_sorted_cmplxids_in_file_agrees_with_expected_order(self, filepath):

        # read from file
        f = open(filepath, 'r')
        filestring = f.read()
        f.close()

        ids_file_sorted = [el for el in filestring.split('\n') if el != '']

        # construct list of ids from container
        ids_cont_sorted = self._CompoundContainer__strandids_sorted + self._CompoundContainer__cmplxids_sorted

        if( ids_file_sorted == ids_cont_sorted):
            return True
        else:
            return False 


    def save_complexes_by_type(self, filepath):

        filestring = "complex type (number of strands, length hybridization sites, dG)\
        \t\tnumber of complexes\n"
        for cmplx_type in self.cmplxs_by_type.keys():
            filestring += "%s\t\t%s\n" %(cmplx_type, self.cmplxs_by_type[cmplx_type])
        filestring = filestring[0:-1]

        # write to file
        f = open(filepath, 'w')
        f.write(filestring)
        f.close()


    def create_map_strandid_to_cmplxids(self):

        self.strandid2cmplxid = {}

        for nc in self.cmplxs.keys():
            for cmplx in self.cmplxs[nc]:
                for strand in cmplx.strands:
                    strand_id = '_'.join(strand.id.split('_')[0:2])
                    if(not strand_id in self.strandid2cmplxid):
                        self.strandid2cmplxid[strand_id] = [cmplx.id]
                    else:
                        self.strandid2cmplxid[strand_id].append(cmplx.id)
    

    def create_map_cmplxid_to_cmplxindexflat(self):

        self.cmplxid2indexflat = {}

        counter = 0
        for n in self.cmplxid2index.keys():
            for cmplxid in self.cmplxid2index[n]:
                self.cmplxid2indexflat[cmplxid] = self.cmplxid2index[n][cmplxid]+counter
            counter += len(self.cmplxid2index[n])


    def create_map_strandindex_to_cmplxindex_flat(self):

        if(not hasattr(self, 'strandidcmplxid')):
            self.create_map_strandid_to_cmplxids()

        self.strandindex2cmplxindexflat = {}
        
        for strandid in self.strandid2index.keys():
            
            strandindex = self.strandid2index[strandid]

            self.strandindex2cmplxindexflat[strandindex] = []

            for cmplxid in self.strandid2cmplxid[strandid]:

                cmplxindexflat = self.cmplxid2indexflat[cmplxid]
                self.strandindex2cmplxindexflat[strandindex].append(cmplxindexflat)


if __name__=='__main__':

    split_index = sys.argv[1]
    split_size = 13000

    ec = EnergyCalculator('../initial_configurations/energy_parameters.txt')
    cc = CompoundContainer('../initial_configurations/complexes.txt', \
                           is_truncated=True, truncation_type='mismatch')
    cc.group_strands_by_length_and_terminalletters()
    cc.list_all_duplexes() # list duplexes without mismatches
    print("triplexes")
    cc.list_all_nplus1_plexes_no_mismatches(2)
    print("tetraplexes")
    cc.list_all_nplus1_plexes_no_mismatches(3)
    print("pentaplexes")
    cc.list_all_nplus1_plexes_no_mismatches(4,split_index*split_size,\
                                            (split_index+1)*split_size)

    
    
