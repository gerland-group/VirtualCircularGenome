#!/bin/env python3

import re

from global_variables import *
import helper_functions as hf

class Helix:

    def __init__(self, strand_bottom, strand_top, start):

        self.strand_bottom = strand_bottom
        self.strand_top = strand_top
        
        self.start_bottom = 0
        self.start_top = start

        self.end_bottom = self.strand_bottom.l
        self.end_top = self.start_top + self.strand_top.l

        self.start_hyb = max([self.start_top, self.start_bottom])
        self.end_hyb = min([self.end_top, self.end_bottom])

        self.l_all = max([self.end_top, self.end_bottom]) - min([self.start_top, self.start_bottom])
        self.l_hyb = self.end_hyb - self.start_hyb

        self.determine_type()
        self.create_id()
    

    def determine_type(self):

        if( (self.start_top <= 0) and (self.end_top < self.end_bottom) ):
            self.type = 'z'
        elif( (self.start_top <= 0) and (self.end_top >= self.end_bottom) ):
            self.type = 't'
        elif( (self.start_top > 0) and (self.end_top >= self.end_bottom) ):
            self.type = 's'
        elif( (self.start_top > 0) and (self.end_top < self.end_bottom) ):
            self.type = 'tinv'
        else:
            raise ValueError("type of helix cannot be determined")


    def rotate(self):

        if(self.type == 'z'):
            self.rotated = Helix(strand_bottom=self.strand_top, \
                strand_top=self.strand_bottom, start=-(self.l_all-self.strand_top.l))
        
        elif(self.type == 's'):
            self.rotated = Helix(strand_bottom=self.strand_top, \
                strand_top=self.strand_bottom, start=(self.l_all-self.strand_bottom.l))
        
        elif(self.type == 't'):
            self.rotated = Helix(strand_bottom=self.strand_top, \
                strand_top=self.strand_bottom, start=(self.strand_top.l-self.strand_bottom.l+self.start_top))
        
        elif(self.type == 'tinv'):
            self.rotated = Helix(strand_bottom=self.strand_top, \
                strand_top=self.strand_bottom, start=-(self.strand_bottom.l-self.strand_top.l-self.start_top))
            
        self.rotated.id = self.id


    def create_humanreadable_representation(self):

        if(self.start_top < 0):
            self.top_3t5 = '-'.join(re.findall('.', self.strand_top.seq_5t3[::-1]))
            self.bottom_5t3 = " "*2*(-self.start_top) + "-".join(re.findall(\
                '.', self.strand_bottom.seq_5t3))

        else:
            self.bottom_5t3 = '-'.join(re.findall('.', self.strand_bottom.seq_5t3))
            self.top_3t5 = " "*2*(self.start_top) + "-".join(re.findall(\
                '.', self.strand_top.seq_5t3[::-1]))
    
    def print_humanreadable(self):

        if(not hasattr(self, 'bottom_5t3')):
            self.create_humanreadable_representation()
        
        print(self.top_3t5+"\n"+self.bottom_5t3)


    def __str__(self, ):
        
        if(not hasattr(self, 'bottom_5t3')):
            self.create_humanreadable_representation()
        
        return self.top_3t5+"\n"+self.bottom_5t3
    

    def create_id(self):

        if(not hasattr(self, 'bottom_5t3')):
            self.create_humanreadable_representation()
       
        l = (max([len(self.top_3t5), len(self.bottom_5t3)])+1)//2
        top_3t5_loc = self.top_3t5
        bottom_5t3_loc = self.bottom_5t3

        if(len(self.top_3t5)<2*l-1):
            top_3t5_loc += " "*(2*l-1-len(self.top_3t5))
        elif(len(self.bottom_5t3)<2*l-1):
            bottom_5t3_loc += " "*(2*l-1-len(self.bottom_5t3))

        # print("top_3t5_loc: ", top_3t5_loc)
        # print("bottom_3t5_loc: ", bottom_5t3_loc)

        top_numb = int(hf.convert_spacelineAGCT_to_123456(top_3t5_loc[::-1]), \
            base=alphabet+3)
        bottom_numb = int(hf.convert_spacelineAGCT_to_123456(bottom_5t3_loc), \
            base=alphabet+3)

        self.id = "h_%s_%s" %(min([top_numb,bottom_numb]), max([top_numb,bottom_numb]))


    def check_if_perfect_match(self):

        if(not hasattr(self, 'bottom_5t3')):
            self.create_humanreadable_representation()

        for i in range(min([len(self.top_3t5), len(self.bottom_5t3)])):
            l1, l2 = self.bottom_5t3[i], self.top_3t5[i]
            if( (l1 != ' ') and (l1 != '-') and (l2 != ' ') and (l2 != '-')):
                comp = hf.check_complementarity_single_letter(self.bottom_5t3[i], self.top_3t5[i])
                if not comp:
                    return False

        return True
