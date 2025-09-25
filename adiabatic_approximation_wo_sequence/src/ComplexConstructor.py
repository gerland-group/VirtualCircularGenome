#!/bin/env python3

import numpy as np

class ComplexConstructor:

    def __init__(self, l_unique=None, alphabet=None, \
                 L_vcg=None, dL_vcg=None, L_vcg_min=None, L_vcg_max=None, \
                 Lmax_stock=None, Ls=None, \
                 comb_vcg=None, gamma_2m=None, gamma_d=None, energy_lb=None, \
                 include_triplexes=True, include_tetraplexes=False):

        # genome properties
        self.l_unique = l_unique
        self.alphabet = alphabet
        
        # ensemble properties
        self.L_vcg = L_vcg # typical length of strand in VCG
        if (dL_vcg is None) and (not L_vcg_min is None) and (not L_vcg_max is None) and (Ls is None):
            self.L_vcg_min = L_vcg_min # minimal length of strand in VCG
            if self.L_vcg_min < self.l_unique:
                raise ValueError('invalid input: smallest oligomer in VCG needs to be longer than l_unique')
            self.L_vcg_max = L_vcg_max # maximal length of strand in VCG
        elif (not dL_vcg is None) and (L_vcg_min is None) and (L_vcg_max is None) and (Ls is None):
            self.L_vcg_min = L_vcg-dL_vcg # minimal length of strand in VCG
            self.L_vcg_max = L_vcg+dL_vcg # maximal length of strand in VCG
        elif (dL_vcg is None) and (L_vcg_min is None) and (L_vcg_max is None) and (not Ls is None):
            self.Ls = np.asarray(Ls)
            self.L2indexss = {L:i for i, L in enumerate(self.Ls)}
            self.L_vcg_min = min([L for L in self.Ls if L >= self.l_unique])
            self.L_vcg_max = max([L for L in self.Ls if L >= self.l_unique])

        self.Lmax_stock = Lmax_stock # maximal length included in feeding stock
        if self.Lmax_stock >= self.l_unique:
            raise ValueError('invalid input: Lmax_stock needs to be shorter than l_unique')
        self.comb_vcg = comb_vcg # cominatoric prefactor of number of strands for fixed length in VCG
        if Ls is None:
            self.list_lengths() # list all length appearing in the ensemble
        
        # energies
        self.gamma_2m = gamma_2m
        self.gamma_d = gamma_d
        self.energy_lb = energy_lb # lower bound for the total hybridization energy

        # construct combinatorics and energies for complexes up to triplexes
        self.construct_combinatorics_and_energies_single_strands()
        self.construct_combinatorics_and_energies_duplexes()
        self.include_triplexes = include_triplexes
        if self.include_triplexes:
            self.construct_combinatorics_and_energies_triplexes()
        self.include_tetraplexes = include_tetraplexes
        if self.include_tetraplexes:
            self.construct_combinatorics_and_energies_tetraplexes()
        self.index2keycmplx = {value:key for key, value in self.key2indexcmplx.items()}
        self.construct_map_indexcmplx2indicesstrands_and_indexstrand2indicescmplxs()
        self.construct_array_indexcmplx2indicesstrands_and_indexstrand2indicescmplxs()

        # compute dissociation constants
        self.convert_energies_to_array()
        self.convert_combinatoric_factors_to_array()
        self.compute_binding_constants()

        # reactions distinguished by genome compatibility, 
        # length category of educts and template and product
        self.identify_reaction_types_cvflvs_all_complexes()

        # reactions distinguished by the oligos that are consumed ("Consumed Oligos")
        self.identify_reaction_types_co_all_complexes()

        # reactions distinguished by the oligos that are consumed ("Consumed Oligos")
        # via ligation with a feeding stock oligomer ("primer extension")
        self.identify_reaction_types_co_fs_all_complexes()

        # reactions distinguished by the length of the oligomers that participate
        # ("Oligomer Length")
        self.identify_reaction_types_ol_all_complexes()

        # reactions distinguished by the length of the hybridization site
        # ("Hybridization Length")
        self.identify_reaction_types_hl_all_complexes()

        # unreactive complexes distinguished length of oligomers
        self.identify_inert_types_all_complexes()
            
    
    def list_lengths(self):

        self.Ls = []
        self.L2indexss = {}

        counter = 0
        for L in range(1, self.Lmax_stock+1):
            self.Ls.append(L)
            self.L2indexss[L] = counter
            counter += 1
        
        for L in range(self.L_vcg_min, self.L_vcg_max+1):
            self.Ls.append(L)
            self.L2indexss[L] = counter
            counter += 1

        self.Ls = np.asarray(self.Ls)


    def construct_combinatorics_and_energies_single_strands(self):

        self.key2indexcmplx = {}
        
        # combinatoric factor
        self.combs = []

        # energy (binding affinity) of complex
        self.energies = []

        counter = 0
        for l in self.Ls:
            self.key2indexcmplx[((l,),())] = counter
            if l < self.l_unique:
                self.combs.append(self.alphabet**l)
            elif l >= self.l_unique:
                self.combs.append(self.comb_vcg)
            self.energies.append(0)
            counter += 1
        

    def identify_overlap_individual_duplex(self, i, Lup, Ldown):
        return min(Ldown, i+Lup) - max(i,0)

    
    def check_status_dangling_ends_individual_duplex(self, i, Lup, Ldown):
        dl = 0 if max(i,0)-min(i,0) == 0 else 1
        dr = 0 if max(Ldown,i+Lup)-min(Ldown,i+Lup) == 0 else 1
        return dl, dr
    

    def compute_combinatoric_factor_and_energy_individual_duplex(self, i, Lup, Ldown):

        Lo = self.identify_overlap_individual_duplex(i, Lup, Ldown)
        dl, dr = self.check_status_dangling_ends_individual_duplex(i, Lup, Ldown)
        
        index_up = self.key2indexcmplx[((Lup,),())]
        index_down = self.key2indexcmplx[((Ldown,),())]
        comb_up = self.combs[index_up]
        comb_down = self.combs[index_down]

        # correct symmetry because we assume ordering for Ldown >= Lup
        if Lup != Ldown:
            if (Lo < self.l_unique) and (Lup > Lo):
                comb = int(comb_down * comb_up/self.alphabet**Lo)
            else:
                comb = int(comb_down)
        elif Lup == Ldown:
            if (Lo < self.l_unique) and (Lup > Lo):
                comb = int(1/2 * comb_down * comb_up/self.alphabet**Lo)
            else:
                comb = int(1/2 * comb_down)

        energy = self.gamma_2m*(Lo-1) + self.gamma_d*(dl+dr)
        if (not self.energy_lb is None) and (energy < self.energy_lb):
            energy = self.energy_lb
        
        return comb, energy


    def construct_combinatorics_and_energies_duplexes(self):

        counter = self.key2indexcmplx[next(reversed(self.key2indexcmplx))]+1
        for Ldown in self.Ls:
            for Lup in self.Ls:
                if (Ldown >= Lup) and (Lup != 1 or Ldown != 1):

                    for i in range(-(Lup-1), Ldown):

                        comb, energy = \
                            self.compute_combinatoric_factor_and_energy_individual_duplex(\
                            i, Lup, Ldown)
                        self.combs.append(comb)
                        self.energies.append(energy)
                        self.key2indexcmplx[((Ldown,Lup),(i,))] = counter
                        counter += 1

    
    def identify_overlap_individual_triplex(self, i, j, Lup1, Lup2, Ldown):
        Lo1 = i+Lup1-max(i,0)
        Lo2 = min(Ldown, j+Lup2)-j
        return Lo1, Lo2


    def check_status_dangling_ends_individual_triplex(self, i, j, Lup1, Lup2, Ldown):
        dl = 0 if max(i,0)-min(i,0) == 0 else 1
        dm = 0 if i+Lup1 == j else 1
        dr = 0 if max(Ldown, j+Lup2)-min(Ldown, j+Lup2) == 0 else 1
        return dl, dm, dr


    def compute_combinatoric_factor_and_energy_individual_triplex(self, i, j, \
            Lup1, Lup2, Ldown):
        
        Lo1, Lo2 = self.identify_overlap_individual_triplex(i, j, Lup1, Lup2, Ldown)
        dl, dm, dr = self.check_status_dangling_ends_individual_triplex(i, j, \
            Lup1, Lup2, Ldown)
        
        index_up1 = self.key2indexcmplx[((Lup1,),())]
        index_up2 = self.key2indexcmplx[((Lup2,),())]
        index_down = self.key2indexcmplx[((Ldown,),())]
        comb_up1 = self.combs[index_up1]
        comb_up2 = self.combs[index_up2]
        comb_down = self.combs[index_down]

        comb = comb_down

        if (Lo1 < self.l_unique) and (Lup1 > Lo1):
            comb *= comb_up1/(self.alphabet**Lo1)
        
        if (Lo2 < self.l_unique) and (Lup2 > Lo2):
            comb *= comb_up2/(self.alphabet**Lo2)
        
        energy = self.gamma_2m*(Lo1+Lo2-2) + self.gamma_2m*(1-dm) \
            + self.gamma_d*(dl+dr+2*dm)
        if (not self.energy_lb is None) and (energy < self.energy_lb):
            energy = self.energy_lb
        
        return comb, energy


    def construct_combinatorics_and_energies_triplexes(self):

        counter = self.key2indexcmplx[next(reversed(self.key2indexcmplx))]+1
        for Ldown in self.Ls:
            for Lup1 in self.Ls:
                for Lup2 in self.Ls:
                    if (Ldown > 1):

                        for i in range(-(Lup1-1), Ldown-Lup1):
                            for j in range(i+Lup1, Ldown):

                                comb, energy = \
                                    self.compute_combinatoric_factor_and_energy_individual_triplex(\
                                    i, j, Lup1, Lup2, Ldown)
                                self.combs.append(comb)
                                self.energies.append(energy)
                                self.key2indexcmplx[((Ldown,Lup1,Lup2),(i,j))] = counter
                                counter += 1

    
    def identify_overlap_individual_tetraplex_1v3(self, i, j, ktop, \
                                                  Lup1, Lup2, Lup3, Ldown):
        Lo1 = i+Lup1-max(0,i)
        Lo2 = Lup2
        Lo3 = min(Ldown,ktop+Lup3)-ktop
        return Lo1, Lo2, Lo3


    def identify_overlap_individual_tetraplex_2v2_ztype(self, i, j, kbot, \
                                                        Lup1, Lup2, Ldown1, Ldown2):
        Lo1 = i+Lup1-max(0,i)
        Lo2 = Ldown1-j
        Lo3 = min(j+Lup2,kbot+Ldown2)-kbot
        return Lo1, Lo2, Lo3
    

    def identify_overlap_individual_tetraplex_2v2_stype(self, i, j, kbot, \
                                                        Lup1, Lup2, Ldown1, Ldown2):
        Lo1 = Ldown1-max(0,i)
        Lo2 = i+Lup1-kbot
        Lo3 = min(kbot+Ldown2,j+Lup2)-j
        return Lo1, Lo2, Lo3
    

    def check_status_dangling_ends_individual_tetraplex_1v3(self, i, j, ktop, \
                                                            Lup1, Lup2, Lup3, Ldown):
        dll = 0 if i == 0 else 1
        dml = 0 if i+Lup1 == j else 1
        dmr = 0 if j+Lup2 == ktop else 1
        drr = 0 if ktop+Lup3 == Ldown else 1
        return dll, dml, dmr, drr
    

    def check_status_dangling_ends_individual_tetraplex_2v2_ztype(self, i, j, kbot, \
                                                                  Lup1, Lup2, Ldown1, Ldown2):
        dl = 0 if i== 0 else 1
        dm1 = 0 if i+Lup1 == j else 1
        dm2 = 0 if Ldown1 == kbot else 1
        dr = 0 if j+Lup2 == kbot+Ldown2 else 1
        return dl, dm1, dm2, dr
    

    def check_status_dangling_ends_individual_tetraplex_2v2_stype(self, i, j, kbot, \
                                                                  Lup1, Lup2, Ldown1, Ldown2):
        dl = 0 if i == 0 else 1
        dm1 = 0 if kbot == Ldown1 else 1
        dm2 = 0 if i+Lup1 == j else 1
        dr = 0 if kbot+Ldown2 == j+Lup2 else 1
        return dl, dm1, dm2, dr


    def compute_combinatoric_factor_and_energy_individual_tetraplex_1v3(self, \
            i, j, ktop, Lup1, Lup2, Lup3, Ldown):
        
        Lo1, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_1v3(\
            i, j, ktop, Lup1, Lup2, Lup3, Ldown)
        
        if Lo1 < 0 or Lo2 < 0 or Lo3 < 0:
            raise ValueError(f'problem for (({Ldown,Lup1,Lup2,Lup3}),({i},{j},{ktop},None))'\
                             +f'Lo1: {Lo1}, Lo2: {Lo2}, Lo3: {Lo3}')
        
        dll, dml, dmr, drr = self.check_status_dangling_ends_individual_tetraplex_1v3(\
            i, j, ktop, Lup1, Lup2, Lup3, Ldown)
        
        index_up1 = self.key2indexcmplx[((Lup1,),())]
        index_up2 = self.key2indexcmplx[((Lup2,),())]
        index_up3 = self.key2indexcmplx[((Lup3,),())]
        index_down = self.key2indexcmplx[((Ldown,),())]
        comb_up1 = self.combs[index_up1]
        comb_up2 = self.combs[index_up2]
        comb_up3 = self.combs[index_up3]
        comb_down = self.combs[index_down]

        comb = comb_down
        if (Lo1 < self.l_unique) and (Lup1 > Lo1):
            comb *= comb_up1/(self.alphabet**Lo1)
        if (Lo2 < self.l_unique) and (Lup2 > Lo2):
            comb *= comb_up2/(self.alphabet**Lo2)
        if (Lo3 < self.l_unique) and (Lup3 > Lo3):
            comb *= comb_up3/(self.alphabet**Lo3)
        
        energy = self.gamma_2m*(Lo1+Lo2+Lo3-3) + self.gamma_2m*(1-dml) \
                 + self.gamma_2m*(1-dmr) + self.gamma_d*(dll+drr+2*dml+2*dmr)
        if (not self.energy_lb is None) and (energy < self.energy_lb):
            energy = self.energy_lb
        
        return comb, energy
    

    def compute_combinatoric_factor_and_energy_individual_tetraplex_2v2_ztype(self, \
            i, j, kbot, Lup1, Lup2, Ldown1, Ldown2, symm):
        
        Lo1, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_2v2_ztype(\
            i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
        
        if Lo1 < 0 or Lo2 < 0 or Lo3 < 0:
            raise ValueError(f'problem for (({Ldown1,Lup1,Lup2,Ldown2}),({i},{j},None,{kbot}))'\
                             +f'Lo1: {Lo1}, Lo2: {Lo2}, Lo3: {Lo3}')
        
        dl, dm1, dm2, dr = self.check_status_dangling_ends_individual_tetraplex_2v2_ztype(\
            i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
        
        index_up1 = self.key2indexcmplx[((Lup1,),())]
        index_up2 = self.key2indexcmplx[((Lup2,),())]
        index_down1 = self.key2indexcmplx[((Ldown1,),())]
        index_down2 = self.key2indexcmplx[((Ldown2,),())]
        comb_up1 = self.combs[index_up1]
        comb_up2 = self.combs[index_up2]
        comb_down1 = self.combs[index_down1]
        comb_down2 = self.combs[index_down2]

        comb = comb_down1
        if (Lo1 < self.l_unique) and (Lup1 > Lo1):
            comb *= comb_up1/(self.alphabet**Lo1)
        if (Lo2 < self.l_unique) and (Lup2 > Lo2):
            comb *= comb_up2/(self.alphabet**Lo2)
        if (Lo3 < self.l_unique) and (Ldown2 > Lo3):
            comb *= comb_down2/(self.alphabet**Lo3)
        
        if symm:
            comb *= 1/2
        
        energy = self.gamma_2m*(Lo1+Lo2+Lo3-3) + self.gamma_2m*(1-dm1) \
                 + self.gamma_2m*(1-dm2) + self.gamma_d*(dl+dr+2*dm1+2*dm2)
        if (not self.energy_lb is None) and (energy < self.energy_lb):
            energy = self.energy_lb
        
        return comb, energy
    

    def compute_combinatoric_factor_and_energy_individual_tetraplex_2v2_stype(self, \
            i, j, kbot, Lup1, Lup2, Ldown1, Ldown2, symm):
        
        Lo1, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_2v2_stype(\
            i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
        
        if Lo1 < 0 or Lo2 < 0 or Lo3 < 0:
            raise ValueError(f'problem for (({Ldown1,Lup1,Lup2,Ldown2}),({i},{j},None,{kbot}))'\
                             +f'Lo1: {Lo1}, Lo2: {Lo2}, Lo3: {Lo3}')
        
        dl, dm1, dm2, dr = self.check_status_dangling_ends_individual_tetraplex_2v2_stype(\
            i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
        
        index_up1 = self.key2indexcmplx[((Lup1,),())]
        index_up2 = self.key2indexcmplx[((Lup2,),())]
        index_down1 = self.key2indexcmplx[((Ldown1,),())]
        index_down2 = self.key2indexcmplx[((Ldown2,),())]
        comb_up1 = self.combs[index_up1]
        comb_up2 = self.combs[index_up2]
        comb_down1 = self.combs[index_down1]
        comb_down2 = self.combs[index_down2]

        comb = comb_down1
        if (Lo1 < self.l_unique) and (Lup1 > Lo1):
            comb *= comb_up1/(self.alphabet**Lo1)
        if (Lo2 < self.l_unique) and (Ldown2 > Lo2):
            comb *= comb_down2/(self.alphabet**Lo2)
        if (Lo3 < self.l_unique) and (Lup2 > Lo3):
            comb *= comb_up2/(self.alphabet**Lo3)
        
        if symm:
            comb *= 1/2

        energy = self.gamma_2m*(Lo1+Lo2+Lo3-3) + self.gamma_2m*(1-dm1) + self.gamma_2m*(1-dm2) \
                 + self.gamma_d*(dl+dr+2*dm1+2*dm2)
        if (not self.energy_lb is None) and (energy < self.energy_lb):
            energy = self.energy_lb
        
        return comb, energy


    def construct_rotated_indices_tetraplex_2v2(self, i, j, kbot, \
                                                Lup1, Lup2, Ldown1, Ldown2):

        js_in = np.array([0,i,j,kbot])
        js_in_shifted = js_in-min([i,0])
        l_total = np.max([js_in_shifted[0]+Ldown1+Ldown2,js_in_shifted[1]+Lup1+Lup2])
        js_out_shifted = np.array([l_total-js_in_shifted[2]-Lup2, \
                                   l_total-js_in_shifted[3]-Ldown2, \
                                   l_total-js_in_shifted[0]-Ldown1, \
                                   l_total-js_in_shifted[1]-Lup1])
        js_out = js_out_shifted-js_out_shifted[0]
        return js_out[1:]


    def choose_rotated_vs_unrotated_key_2v2(self, i, j, kbot, \
                                            Lup1, Lup2, Ldown1, Ldown2):

        irot, jrot, kbotrot = self.construct_rotated_indices_tetraplex_2v2(\
            i,j,kbot,Lup1,Lup2,Ldown1,Ldown2)
        Lup1rot = Ldown2
        Lup2rot = Ldown1
        Ldown1rot = Lup2
        Ldown2rot = Lup1

        key = ((Ldown1,Lup1,Lup2,Ldown2),(i,j,None,kbot))
        key_rot = ((Ldown1rot, Lup1rot, Lup2rot, Ldown2rot),(irot,jrot,None,kbotrot))
        symm = True if key == key_rot else False

        if Ldown1 < Lup2:
            Ls = (Ldown1,Lup1,Lup2,Ldown2)
            js = (i,j,None,kbot)
            return (Ls,js), i, j, kbot, Lup1, Lup2, Ldown1, Ldown2, symm
        
        elif Ldown1 > Lup2:
            assert Ldown1rot < Lup2rot
            Ls = (Ldown1rot, Lup1rot, Lup2rot, Ldown2rot)
            js = (irot,jrot,None,kbotrot)
            return (Ls,js), irot, jrot, kbotrot, Lup1rot, Lup2rot, Ldown1rot, Ldown2rot, symm
    
        elif Ldown1 == Lup2:
            
            if Ldown2 > Lup1:
                Ls = (Ldown1,Lup1,Lup2,Ldown2)
                js = (i,j,None,kbot)
                return (Ls,js), i, j, kbot, Lup1, Lup2, Ldown1, Ldown2, symm
            
            elif Ldown2 < Lup1:
                assert Ldown2rot > Lup1rot
                Ls = (Ldown1rot, Lup1rot, Lup2rot, Ldown2rot)
                js = (irot,jrot,None,kbotrot)
                return (Ls,js), irot, jrot, kbotrot, Lup1rot, Lup2rot, Ldown1rot, Ldown2rot, symm

            elif Ldown2 == Lup1:

                sorting = sorted([0,1], key = lambda el: [[i,j,kbot], [irot,jrot,kbotrot]][el])[0]
                Ls = [(Ldown1,Lup1,Lup2,Ldown2),(Ldown1rot,Lup1rot,Lup2rot,Ldown2rot)][sorting]
                js = [(i,j,None,kbot),(irot,jrot,None,kbotrot)][sorting]
                ifinal = [i,irot][sorting]
                jfinal = [j,jrot][sorting]
                kbotfinal = [kbot,kbotrot][sorting]
                Lup1final = [Lup1,Lup1rot][sorting]
                Lup2final = [Lup2,Lup2rot][sorting]
                Ldown1final = [Ldown1,Ldown1rot][sorting]
                Ldown2final = [Ldown2, Ldown2rot][sorting]        
                return (Ls,js), ifinal, jfinal, kbotfinal, \
                       Lup1final, Lup2final, Ldown1final, Ldown2final, symm
        
        else:
            raise ValueError('key cannot be chosen')


    def construct_combinatorics_and_energies_tetraplexes_1v3(self):

        counter = self.key2indexcmplx[next(reversed(self.key2indexcmplx))]+1
        for Ldown in self.Ls:
            if (Ldown > 1):
                for Lup1 in self.Ls:
                    for Lup2 in self.Ls:
                        for Lup3 in self.Ls:

                            for i in range(-(Lup1-1), Ldown-Lup1-Lup2):
                                for j in range(i+Lup1, Ldown-Lup2):
                                    for ktop in range(j+Lup2,Ldown):

                                        comb, energy = \
                                            self.compute_combinatoric_factor_and_energy_individual_tetraplex_1v3(\
                                                i, j, ktop, Lup1, Lup2, Lup3, Ldown)
                                        self.combs.append(comb)
                                        self.energies.append(energy)
                                        self.key2indexcmplx[((Ldown,Lup1,Lup2,Lup3),(i,j,ktop,None))] = counter
                                        counter += 1


    def construct_combinatorics_and_energies_tetraplexes_2v2_ztype(self):

        # tetraplexes of form 2v2 z-type
        counter = self.key2indexcmplx[next(reversed(self.key2indexcmplx))]+1
        for Ldown1 in self.Ls:
                for Lup2 in self.Ls:
                    for Lup1 in self.Ls:
                        for Ldown2 in self.Ls:

                            for i in range(-(Lup1-1),Ldown1-Lup1):
                                for j in range(i+Lup1,Ldown1):
                                    for kbot in range(Ldown1, j+Lup2):
                            
                                        key_final, i_final, j_final, kbot_final, \
                                            Lup1_final, Lup2_final, Ldown1_final, \
                                            Ldown2_final, symm = \
                                        self.choose_rotated_vs_unrotated_key_2v2(\
                                            i,j,kbot,Lup1,Lup2,Ldown1,Ldown2)

                                        if (not key_final in self.key2indexcmplx):
                                            comb, energy = \
                                                self.compute_combinatoric_factor_and_energy_individual_tetraplex_2v2_ztype(\
                                                    i_final, j_final, kbot_final, \
                                                    Lup1_final, Lup2_final, Ldown1_final, \
                                                    Ldown2_final, symm)
                                            self.combs.append(comb)
                                            self.energies.append(energy)
                                            self.key2indexcmplx[key_final] = counter
                                            counter += 1


    def construct_combinatorics_and_energies_tetraplexes_2v2_stype(self):

        # tetraplexes of form 2v2 s-type
        counter = self.key2indexcmplx[next(reversed(self.key2indexcmplx))]+1
        for Ldown1 in self.Ls:
            for Lup2 in self.Ls:
                for Lup1 in self.Ls:
                    for Ldown2 in self.Ls:

                        for i in range(Ldown1-Lup1+1, Ldown1):
                            for kbot in range(Ldown1, i+Lup1):
                                for j in range(i+Lup1,kbot+Ldown2):

                                    key_final, i_final, j_final, kbot_final, \
                                        Lup1_final, Lup2_final, Ldown1_final, \
                                        Ldown2_final, symm = \
                                    self.choose_rotated_vs_unrotated_key_2v2(\
                                        i,j,kbot,Lup1,Lup2,Ldown1,Ldown2)

                                    if (not key_final in self.key2indexcmplx):
                                        comb, energy = \
                                            self.compute_combinatoric_factor_and_energy_individual_tetraplex_2v2_stype(\
                                                i_final, j_final, kbot_final, \
                                                Lup1_final, Lup2_final, Ldown1_final, \
                                                Ldown2_final, symm)
                                        self.combs.append(comb)
                                        self.energies.append(energy)
                                        self.key2indexcmplx[key_final] = counter
                                        counter += 1


    def construct_combinatorics_and_energies_tetraplexes(self):
        self.construct_combinatorics_and_energies_tetraplexes_1v3()
        self.construct_combinatorics_and_energies_tetraplexes_2v2_ztype()
        self.construct_combinatorics_and_energies_tetraplexes_2v2_stype()


    def construct_map_indexcmplx2indicesstrands_and_indexstrand2indicescmplxs(self):

        # point from index of complex to indices of strands incorporated in the complex
        self.indexcmplx2indicesstrands = {}

        # point from strand to all complexes in which the strand is included
        self.indexstrand2indicescmplxs = {}

        for key, indexcmplx in self.key2indexcmplx.items():
            
            indicesstrands_single_cmplx = []
            for L in key[0]:
                
                indexstrand = self.L2indexss[L]
                indicesstrands_single_cmplx.append(indexstrand)

                if not indexstrand in self.indexstrand2indicescmplxs:
                    self.indexstrand2indicescmplxs[indexstrand] = [indexcmplx]
                elif (indexstrand in self.indexstrand2indicescmplxs):
                    self.indexstrand2indicescmplxs[indexstrand].append(indexcmplx)

            self.indexcmplx2indicesstrands[indexcmplx] = np.asarray(indicesstrands_single_cmplx)

        for key, value in self.indexstrand2indicescmplxs.items():
            self.indexstrand2indicescmplxs[key] = np.asarray(value)


    def construct_array_indexcmplx2indicesstrands_and_indexstrand2indicescmplxs(self):

        if not hasattr(self, 'indexstrand2indicexcmplxs'):
            self.construct_map_indexcmplx2indicesstrands_and_indexstrand2indicescmplxs()

        n_strands_max = len(next(reversed(self.key2indexcmplx))[0])
        
        # point from index of complex to indices of strands incorporated in the complex
        self.indexcmplx2indicesstrands_arr = (-33)*np.ones((len(self.key2indexcmplx), \
                                                            n_strands_max), dtype=int)

        for key, indexcmplx in self.key2indexcmplx.items():
            for i, L in enumerate(key[0]):
                indexstrand = self.L2indexss[L]
                self.indexcmplx2indicesstrands_arr[indexcmplx][i] = indexstrand

        # point from strand to all complexes in which the strand is included
        n_cmplxs_max = max([len(el) for el in self.indexstrand2indicescmplxs.values()])
        self.indexstrand2indicescmplxs_arr = (-33)*np.ones((len(self.L2indexss), \
                                                            n_cmplxs_max), dtype=int)
        for indexstrand, indicescmplxs in self.indexstrand2indicescmplxs.items():
            self.indexstrand2indicescmplxs_arr[indexstrand][0:len(indicescmplxs)] = indicescmplxs
    

    def add_ligation_site_to_reaction_type_dictionaries(self, icmplx, \
                                                        Leduct1, Leduct2, Ltemp, \
                                                        Lovlp1, Lovlp2, \
                                                        to_cvflvs, to_cvfol, to_cvfolp):

        stock_educt1 = 1 if Leduct1 >= self.l_unique else 0
        stock_educt2 = 1 if Leduct2 >= self.l_unique else 0
        stock_temp = 1 if Ltemp >= self.l_unique else 0
        stock_prod = 1 if Leduct1+Leduct2 >= self.l_unique else 0
        number_added_nucleotides = min([Leduct1, Leduct2])

        comb_prod = self.alphabet**(Leduct1+Leduct2) \
            if (Leduct1+Leduct2) < self.l_unique else self.comb_vcg
        comb_educt1 = self.alphabet**Leduct1 if Leduct1 < self.l_unique else self.comb_vcg
        comb_educt2 = self.alphabet**Leduct2 if Leduct2 < self.l_unique else self.comb_vcg
        comb_overlap1 = self.alphabet**Lovlp1 if Lovlp1 < self.l_unique else self.comb_vcg
        comb_overlap2 = self.alphabet**Lovlp2 if Lovlp2 < self.l_unique else self.comb_vcg
        comb_overlapfull = self.alphabet**(Lovlp1+Lovlp2) \
            if (Lovlp1+Lovlp2) < self.l_unique else self.comb_vcg
        
        probability_genome_compatibility = \
                comb_prod/(comb_educt1/comb_overlap1 * comb_overlapfull * comb_educt2/comb_overlap2)
            
        properties_correct = (stock_educt1, stock_educt2, stock_temp, stock_prod, 1)
        if properties_correct in self.properties_2_react_type_cvflvs_index \
            and probability_genome_compatibility != 0.:
            
            if to_cvflvs:
                react_type_index = self.properties_2_react_type_cvflvs_index[\
                    properties_correct]
                self.react_type_cvflvs_index_2_cmplx_indices[react_type_index].append(icmplx)
                self.react_type_cvflvs_index_2_cmplx_weights[react_type_index].append(\
                    probability_genome_compatibility)
                self.react_type_cvflvs_index_2_added_nucleotides[react_type_index].append(\
                    number_added_nucleotides)
            
            if to_cvfol:
                react_type = str(self.index2keycmplx[icmplx][0]) + "_" \
                             + f"({Ltemp}, {min([Leduct1,Leduct2])}, {max([Leduct1,Leduct2])})" \
                             + "_c"
                if react_type in self.react_type_cvfol_2_index:
                    react_type_index = self.react_type_cvfol_2_index[react_type]
                elif not react_type in self.react_type_cvfol_2_index:
                    if len(self.react_type_cvfol_2_index) == 0:
                        react_type_index = 0
                    elif len(self.react_type_cvfol_2_index) != 0:
                        last_index = self.react_type_cvfol_2_index[next(reversed(self.react_type_cvfol_2_index))]
                        react_type_index = last_index+1
                    self.react_type_cvfol_2_index[react_type] = react_type_index
                    self.react_type_cvfol_index_2_cmplx_indices[react_type_index] = []
                    self.react_type_cvfol_index_2_cmplx_weights[react_type_index] = []
                    self.react_type_cvfol_index_2_added_nucleotides[react_type_index] = []
                self.react_type_cvfol_index_2_cmplx_indices[react_type_index].append(icmplx)
                self.react_type_cvfol_index_2_cmplx_weights[react_type_index].append(\
                    probability_genome_compatibility)
                self.react_type_cvfol_index_2_added_nucleotides[react_type_index].append(\
                    number_added_nucleotides)
            
            if to_cvfolp:
                react_type = f"({Ltemp}, {min([Leduct1,Leduct2])}, {max([Leduct1,Leduct2])})" \
                             + "_c"
                if react_type in self.react_type_cvfolp_2_index:
                    react_type_index = self.react_type_cvfolp_2_index[react_type]
                elif not react_type in self.react_type_cvfolp_2_index:
                    if len(self.react_type_cvfolp_2_index) == 0:
                        react_type_index = 0
                    elif len(self.react_type_cvfolp_2_index) != 0:
                        last_index = self.react_type_cvfolp_2_index[next(reversed(self.react_type_cvfolp_2_index))]
                        react_type_index = last_index+1
                    self.react_type_cvfolp_2_index[react_type] = react_type_index
                    self.react_type_cvfolp_index_2_cmplx_indices[react_type_index] = []
                    self.react_type_cvfolp_index_2_cmplx_weights[react_type_index] = []
                    self.react_type_cvfolp_index_2_added_nucleotides[react_type_index] = []
                self.react_type_cvfolp_index_2_cmplx_indices[react_type_index].append(icmplx)
                self.react_type_cvfolp_index_2_cmplx_weights[react_type_index].append(\
                    probability_genome_compatibility)
                self.react_type_cvfolp_index_2_added_nucleotides[react_type_index].append(\
                    number_added_nucleotides)
                
        properties_false = (stock_educt1, stock_educt2, stock_temp, stock_prod, 0)
        if properties_false in self.properties_2_react_type_cvflvs_index \
            and 1-probability_genome_compatibility != 0.:
        
            if to_cvflvs:
                react_type_index = self.properties_2_react_type_cvflvs_index[\
                    properties_false]
                self.react_type_cvflvs_index_2_cmplx_indices[react_type_index].append(icmplx)
                self.react_type_cvflvs_index_2_cmplx_weights[react_type_index].append(\
                    1-probability_genome_compatibility)
                self.react_type_cvflvs_index_2_added_nucleotides[react_type_index].append(\
                    number_added_nucleotides)
            
            if to_cvfol:
                react_type = str(self.index2keycmplx[icmplx][0]) + "_" \
                             + f"({Ltemp}, {min([Leduct1,Leduct2])}, {max([Leduct1,Leduct2])})" \
                             + "_f"
                if react_type in self.react_type_cvfol_2_index:
                    react_type_index = self.react_type_cvfol_2_index[react_type]
                elif not react_type in self.react_type_cvfol_2_index:
                    if len(self.react_type_cvfol_2_index) == 0:
                        react_type_index = 0
                    elif len(self.react_type_cvfol_2_index) != 0:
                        last_index = self.react_type_cvfol_2_index[next(reversed(self.react_type_cvfol_2_index))]
                        react_type_index = last_index+1
                    self.react_type_cvfol_2_index[react_type] = react_type_index
                    self.react_type_cvfol_index_2_cmplx_indices[react_type_index] = []
                    self.react_type_cvfol_index_2_cmplx_weights[react_type_index] = []
                    self.react_type_cvfol_index_2_added_nucleotides[react_type_index] = []
                self.react_type_cvfol_index_2_cmplx_indices[react_type_index].append(icmplx)
                self.react_type_cvfol_index_2_cmplx_weights[react_type_index].append(\
                    1-probability_genome_compatibility)
                self.react_type_cvfol_index_2_added_nucleotides[react_type_index].append(\
                    number_added_nucleotides)
            
            if to_cvfolp:
                react_type = f"({Ltemp}, {min([Leduct1,Leduct2])}, {max([Leduct1,Leduct2])})" \
                             + "_f"
                if react_type in self.react_type_cvfolp_2_index:
                    react_type_index = self.react_type_cvfolp_2_index[react_type]
                elif not react_type in self.react_type_cvfolp_2_index:
                    if len(self.react_type_cvfolp_2_index) == 0:
                        react_type_index = 0
                    elif len(self.react_type_cvfolp_2_index) != 0:
                        last_index = self.react_type_cvfolp_2_index[next(reversed(self.react_type_cvfolp_2_index))]
                        react_type_index = last_index+1
                    self.react_type_cvfolp_2_index[react_type] = react_type_index
                    self.react_type_cvfolp_index_2_cmplx_indices[react_type_index] = []
                    self.react_type_cvfolp_index_2_cmplx_weights[react_type_index] = []
                    self.react_type_cvfolp_index_2_added_nucleotides[react_type_index] = []
                self.react_type_cvfolp_index_2_cmplx_indices[react_type_index].append(icmplx)
                self.react_type_cvfolp_index_2_cmplx_weights[react_type_index].append(\
                    1-probability_genome_compatibility)
                self.react_type_cvfolp_index_2_added_nucleotides[react_type_index].append(\
                    number_added_nucleotides)


    def identify_reaction_type_cvflvs_single_triplex(self, icmplx, cmplx_key, \
                                                     to_cvflvs=False, to_cvfol=False, \
                                                     to_cvfolp=False):

        Ldown = cmplx_key[0][0]
        Lup1 = cmplx_key[0][1]
        Lup2 = cmplx_key[0][2]

        i = cmplx_key[1][0]
        j = cmplx_key[1][1]

        if i + Lup1 == j: # oligomers are adjacent to each other
            
            Lo1, Lo2 = self.identify_overlap_individual_triplex(\
                i, j, Lup1, Lup2, Ldown)
            self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, \
                Leduct1=Lup1, Leduct2=Lup2, Ltemp=Ldown, Lovlp1=Lo1, Lovlp2=Lo2, \
                to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
    

    def identify_reaction_type_cvflvs_single_tetraplex_1v3(self, icmplx, cmplx_key, \
                                                           to_cvflvs=False, to_cvfol=False, \
                                                           to_cvfolp=False):

        Ldown, Lup1, Lup2, Lup3 = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]

        if i+Lup1==j:
            Lo1, Lo2, _ = self.identify_overlap_individual_tetraplex_1v3(\
                i, j, ktop, Lup1, Lup2, Lup3, Ldown)
            self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, \
                Leduct1=Lup1, Leduct2=Lup2, Ltemp=Ldown, Lovlp1=Lo1, Lovlp2=Lo2, \
                to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
            
        if j+Lup2==ktop:
            _, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_1v3(\
                i, j, ktop, Lup1, Lup2, Lup3, Ldown)
            self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, 
                Leduct1=Lup2, Leduct2=Lup3, Ltemp=Ldown, Lovlp1=Lo2, Lovlp2=Lo3, \
                to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
    

    def identify_reaction_type_cvflvs_single_tetraplex_2v2(self, icmplx, cmplx_key, \
                                                           to_cvflvs=False, to_cvfol=False, \
                                                           to_cvfolp=False):

        Ldown1, Lup1, Lup2, Ldown2 = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if i+Lup1 > Ldown1:
            # s-type

            if i+Lup1==j:
                _, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_2v2_stype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, \
                    Leduct1=Lup1, Leduct2=Lup2, Ltemp=Ldown2, Lovlp1=Lo2, Lovlp2=Lo3, \
                    to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)

            if Ldown1==kbot:
                Lo1, Lo2, _ = self.identify_overlap_individual_tetraplex_2v2_stype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, \
                    Leduct1=Ldown2, Leduct2=Ldown1, Ltemp=Lup1, Lovlp1=Lo2, Lovlp2=Lo1, \
                    to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
                
        elif i+Lup1 < Ldown1:
            # z-type

            if i+Lup1==j:
                Lo1, Lo2, _ = self.identify_overlap_individual_tetraplex_2v2_ztype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, \
                    Leduct1=Lup1, Leduct2=Lup2, Ltemp=Ldown1, Lovlp1=Lo1, Lovlp2=Lo2, \
                    to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
            
            if Ldown1==kbot:
                _, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_2v2_ztype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                self.add_ligation_site_to_reaction_type_dictionaries(icmplx=icmplx, \
                    Leduct1=Ldown2, Leduct2=Ldown1, Ltemp=Lup2, Lovlp1=Lo3, Lovlp2=Lo2,
                    to_cvflvs=to_cvflvs, to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
        
    
    def identify_reaction_type_cvflvs_single_tetraplex(self, icmplx, cmplx_key, \
                                                       to_cvflvs=False, to_cvfol=False, \
                                                       to_cvfolp=False):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_reaction_type_cvflvs_single_tetraplex_1v3(\
                icmplx=icmplx, cmplx_key=cmplx_key, to_cvflvs=to_cvflvs, \
                to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_reaction_type_cvflvs_single_tetraplex_2v2(\
                icmplx=icmplx, cmplx_key=cmplx_key, to_cvflvs=to_cvflvs, \
                to_cvfol=to_cvfol, to_cvfolp=to_cvfolp)
        
        else:
            raise ValueError('unexpected key for tetraplex')
    

    def identify_reaction_types_cvflvs_all_complexes(self):

        self.react_type_cvflvs_2_index = {'educts_fed_fed__temp_fed__prod_fed__correct':0, \
                                          'educts_fed_fed__temp_fed__prod_vcg__correct':1, \
                                          'educts_fed_fed__temp_fed__prod_vcg__false':2, \
                                          'educts_fed_fed__temp_vcg__prod_fed__correct':3, \
                                          'educts_fed_fed__temp_vcg__prod_vcg__correct':4, \
                                          'educts_fed_fed__temp_vcg__prod_vcg__false':5, \
                                          
                                          'educts_fed_vcg__temp_fed__prod_vcg__correct':6, \
                                          'educts_fed_vcg__temp_fed__prod_vcg__false':7, \
                                          'educts_fed_vcg__temp_vcg__prod_vcg__correct':8, \
                                          'educts_fed_vcg__temp_vcg__prod_vcg__false':9, \
                                          
                                          'educts_vcg_vcg__temp_fed__prod_vcg__correct':10, \
                                          'educts_vcg_vcg__temp_fed__prod_vcg__false':11, \
                                          'educts_vcg_vcg__temp_vcg__prod_vcg__correct':12, \
                                          'educts_vcg_vcg__temp_vcg__prod_vcg__false':13, \
                                          }
        self.index_2_react_type_cvflvs_simpnot = {
            value:key[7] + key[11] + "_" + key[21] + "_" + key[31] + "_" + key[36] \
            for key, value in self.react_type_cvflvs_2_index.items() }
        self.properties_2_react_type_cvflvs_index = {
            (0,0,0,0,1):0, (0,0,0,1,1):1, (0,0,0,1,0):2, (0,0,1,0,1):3, \
            (0,0,1,1,1):4, (0,0,1,1,0):5, \
            (0,1,0,1,1):6, (0,1,0,1,0):7, (0,1,1,1,1):8, (0,1,1,1,0):9, \
            (1,0,0,1,1):6, (1,0,0,1,0):7, (1,0,1,1,1):8, (1,0,1,1,0):9, \
            (1,1,0,1,1):10, (1,1,0,1,0):11, (1,1,1,1,1):12, (1,1,1,1,0):13
        }

        self.react_type_cvflvs_index_2_cmplx_indices = \
            {index:[] for index in self.react_type_cvflvs_2_index.values()}
        self.react_type_cvflvs_index_2_cmplx_weights = \
            {index:[] for index in self.react_type_cvflvs_2_index.values()}
        self.react_type_cvflvs_index_2_added_nucleotides = \
            {index:[] for index in self.react_type_cvflvs_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_cvflvs_single_triplex(\
                    icmplx=icmplx, cmplx_key=cmplxkey, to_cvflvs=True)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_cvflvs_single_tetraplex(\
                    icmplx=icmplx, cmplx_key=cmplxkey, to_cvflvs=True)

        self.react_type_cvflvs_index_2_cmplx_weights = \
            {key:np.asarray(value) \
             for key,value in self.react_type_cvflvs_index_2_cmplx_weights.items()}
        self.react_type_cvflvs_index_2_added_nucleotides = \
            {key:np.asarray(value) \
             for key,value in self.react_type_cvflvs_index_2_added_nucleotides.items()}

    
    def identify_reaction_types_cvfol_all_complexes(self):

        self.react_type_cvfol_2_index = {}
        self.react_type_cvfol_index_2_cmplx_indices = {}
        self.react_type_cvfol_index_2_cmplx_weights = {}
        self.react_type_cvfol_index_2_added_nucleotides = {}

        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_cvflvs_single_triplex(\
                    icmplx=icmplx, cmplx_key=cmplxkey, to_cvfol=True)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_cvflvs_single_tetraplex(\
                    icmplx=icmplx, cmplx_key=cmplxkey, to_cvfol=True)
        
        self.react_type_cvfol_index_2_cmplx_weights = \
            {key:np.asarray(value) \
             for key,value in self.react_type_cvfol_index_2_cmplx_weights.items()}
        self.react_type_cvfol_index_2_added_nucleotides = \
            {key:np.asarray(value) \
             for key,value in self.react_type_cvfol_index_2_added_nucleotides.items()}
        
    
    def identify_reaction_types_cvfolp_all_complexes(self):

        self.react_type_cvfolp_2_index = {}
        self.react_type_cvfolp_index_2_cmplx_indices = {}
        self.react_type_cvfolp_index_2_cmplx_weights = {}
        self.react_type_cvfolp_index_2_added_nucleotides = {}

        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_cvflvs_single_triplex(\
                    icmplx=icmplx, cmplx_key=cmplxkey, to_cvfolp=True)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_cvflvs_single_tetraplex(\
                    icmplx=icmplx, cmplx_key=cmplxkey, to_cvfolp=True)
        
        self.react_type_cvfolp_index_2_cmplx_weights = \
            {key:np.asarray(value) \
             for key,value in self.react_type_cvfolp_index_2_cmplx_weights.items()}
        self.react_type_cvfolp_index_2_added_nucleotides = \
            {key:np.asarray(value) \
             for key,value in self.react_type_cvfolp_index_2_added_nucleotides.items()}
    

    def identify_reaction_type_co_single_triplex(self, icmplx, cmplx_key):

        Lup1 = cmplx_key[0][1]
        Lup2 = cmplx_key[0][2]

        i = cmplx_key[1][0]
        j = cmplx_key[1][1]

        if i + Lup1 == j: # oligomers are adjacent to each other
            rt_index_1 = self.react_type_co_2_index[Lup1]
            self.react_type_co_2_cmplx_indices[rt_index_1].append(icmplx)
            rt_index_2 = self.react_type_co_2_index[Lup2]
            self.react_type_co_2_cmplx_indices[rt_index_2].append(icmplx)


    def identify_reaction_type_co_single_tetraplex_1v3(self, icmplx, cmplx_key):

        _, Lup1, Lup2, Lup3 = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]

        if i+Lup1==j:
            rt_index_1 = self.react_type_co_2_index[Lup1]
            self.react_type_co_2_cmplx_indices[rt_index_1].append(icmplx)
            rt_index_2 = self.react_type_co_2_index[Lup2]
            self.react_type_co_2_cmplx_indices[rt_index_2].append(icmplx)

        if j+Lup2==ktop:
            rt_index_2 = self.react_type_co_2_index[Lup2]
            self.react_type_co_2_cmplx_indices[rt_index_2].append(icmplx)
            rt_index_3 = self.react_type_co_2_index[Lup3]
            self.react_type_co_2_cmplx_indices[rt_index_3].append(icmplx)
    

    def identify_reaction_type_co_single_tetraplex_2v2(self, icmplx, cmplx_key):

        Ldown1, Lup1, Lup2, Ldown2 = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if i+Lup1 == j:
            rt_index_1 = self.react_type_co_2_index[Lup1]
            self.react_type_co_2_cmplx_indices[rt_index_1].append(icmplx)
            rt_index_2 = self.react_type_co_2_index[Lup2]
            self.react_type_co_2_cmplx_indices[rt_index_2].append(icmplx)
        
        if Ldown1 == kbot:
            rt_index_1 = self.react_type_co_2_index[Ldown1]
            self.react_type_co_2_cmplx_indices[rt_index_1].append(icmplx)
            rt_index_2 = self.react_type_co_2_index[Ldown2]
            self.react_type_co_2_cmplx_indices[rt_index_2].append(icmplx)


    def identify_reaction_type_co_single_tetraplex(self, icmplx, cmplx_key):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_reaction_type_co_single_tetraplex_1v3(icmplx, cmplx_key)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_reaction_type_co_single_tetraplex_2v2(icmplx, cmplx_key)
        
        else:
            raise ValueError('unexpected key for tetraplex')
        
    
    def identify_reaction_types_co_all_complexes(self):
        
        self.react_type_co_2_index = {L:i for i, L in enumerate(self.Ls)}
        self.index_2_react_type_co = {value:key for key, value \
                                      in self.react_type_co_2_index.items()}
        self.react_type_co_2_cmplx_indices = \
            {index:[] for index in self.react_type_co_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_co_single_triplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_co_single_tetraplex(icmplx, cmplxkey)


    def identify_reaction_type_co_fs_single_triplex(self, icmplx, cmplx_key):

        Lup1 = cmplx_key[0][1]
        Lup2 = cmplx_key[0][2]

        i = cmplx_key[1][0]
        j = cmplx_key[1][1]

        if (i+Lup1==j) and (Lup1 >= self.l_unique) and (Lup2 < self.l_unique):
            rt_index_1 = self.react_type_co_fs_2_index[Lup1]
            self.react_type_co_fs_2_cmplx_indices[rt_index_1].append(icmplx)
        
        if (i+Lup1==j) and (Lup1 < self.l_unique) and (Lup2 >= self.l_unique):
            rt_index_2 = self.react_type_co_fs_2_index[Lup2]
            self.react_type_co_fs_2_cmplx_indices[rt_index_2].append(icmplx)


    def identify_reaction_type_co_fs_single_tetraplex_1v3(self, icmplx, cmplx_key):

        _, Lup1, Lup2, Lup3 = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]

        if (i+Lup1==j) and (Lup1 >= self.l_unique) and (Lup2 < self.l_unique):
            rt_index_1 = self.react_type_co_fs_2_index[Lup1]
            self.react_type_co_fs_2_cmplx_indices[rt_index_1].append(icmplx)

        if (i+Lup1==j) and (Lup1 < self.l_unique) and (Lup2 >= self.l_unique):
            rt_index_2 = self.react_type_co_fs_2_index[Lup2]
            self.react_type_co_fs_2_cmplx_indices[rt_index_2].append(icmplx)
        
        if (j+Lup2==ktop) and (Lup2 >= self.l_unique) and (Lup3 < self.l_unique):
            rt_index_2 = self.react_type_co_fs_2_index[Lup2]
            self.react_type_co_fs_2_cmplx_indices[rt_index_2].append(icmplx)

        if (j+Lup2==ktop) and (Lup2 < self.l_unique) and (Lup3 >= self.l_unique):
            rt_index_3 = self.react_type_co_fs_2_index[Lup3]
            self.react_type_co_fs_2_cmplx_indices[rt_index_3].append(icmplx)
        
    
    def identify_reaction_type_co_fs_single_tetraplex_2v2(self, icmplx, cmplx_key):

        Ldown1, Lup1, Lup2, Ldown2 = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if (i+Lup1 == j) and (Lup1 >= self.l_unique) and (Lup2 < self.l_unique):
            rt_index_1 = self.react_type_co_fs_2_index[Lup1]
            self.react_type_co_fs_2_cmplx_indices[rt_index_1].append(icmplx)
        
        if (i+Lup1 == j) and (Lup1 < self.l_unique) and (Lup2 >= self.l_unique):
            rt_index_2 = self.react_type_co_fs_2_index[Lup2]
            self.react_type_co_fs_2_cmplx_indices[rt_index_2].append(icmplx)
        
        if (Ldown1 == kbot) and (Ldown1 >= self.l_unique) and (Ldown2 < self.l_unique):
            rt_index_1 = self.react_type_co_fs_2_index[Ldown1]
            self.react_type_co_fs_2_cmplx_indices[rt_index_1].append(icmplx)
        
        if (Ldown1 == kbot) and (Ldown1 < self.l_unique) and (Ldown2 >= self.l_unique):
            rt_index_2 = self.react_type_co_fs_2_index[Ldown2]
            self.react_type_co_fs_2_cmplx_indices[rt_index_2].append(icmplx)
    

    def identify_reaction_type_co_fs_single_tetraplex(self, icmplx, cmplx_key):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_reaction_type_co_fs_single_tetraplex_1v3(icmplx, cmplx_key)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_reaction_type_co_fs_single_tetraplex_2v2(icmplx, cmplx_key)
        
        else:
            raise ValueError('unexpected key for tetraplex')
    

    def identify_reaction_types_co_fs_all_complexes(self):

        self.react_type_co_fs_2_index = {L:i for i, L in enumerate(\
                                         self.Ls[self.Ls>=self.l_unique])}
        self.index_2_react_type_co_fs = {value:key for key, value \
                                         in self.react_type_co_fs_2_index.items()}
        self.react_type_co_fs_2_cmplx_indices = \
            {index:[] for index in self.react_type_co_fs_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_co_fs_single_triplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_co_fs_single_tetraplex(icmplx, cmplxkey)

    
    def identify_reaction_type_ol_single_triplex(self, icmplx, cmplx_key):

        Lup1, Lup2 = cmplx_key[0][1:]
        i, j = cmplx_key[1]
        
        if (i+Lup1==j):
            rt_index = self.react_type_ol_2_index[cmplx_key[0]]
            self.react_type_ol_2_cmplx_indices[rt_index].append(icmplx)

    
    def identify_reaction_type_ol_single_tetraplex_1v3(self, icmplx, cmplx_key):

        _, Lup1, Lup2, Lup3 = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]

        if (i+Lup1==j):
            rt_index = self.react_type_ol_2_index[cmplx_key[0]]
            self.react_type_ol_2_cmplx_indices[rt_index].append(icmplx)
        
        if (j+Lup2==ktop):
            rt_index = self.react_type_ol_2_index[cmplx_key[0]]
            self.react_type_ol_2_cmplx_indices[rt_index].append(icmplx)


    def identify_reaction_type_ol_single_tetraplex_2v2(self, icmplx, cmplx_key):

        Ldown1, Lup1, Lup2, Ldown2 = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if (i+Lup1 == j):
            rt_index = self.react_type_ol_2_index[cmplx_key[0]]
            self.react_type_ol_2_cmplx_indices[rt_index].append(icmplx)
        
        if (Ldown1 == kbot):
            rt_index = self.react_type_ol_2_index[cmplx_key[0]]
            self.react_type_ol_2_cmplx_indices[rt_index].append(icmplx)
    

    def identify_reaction_type_ol_single_tetraplex(self, icmplx, cmplx_key):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_reaction_type_ol_single_tetraplex_1v3(icmplx, cmplx_key)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_reaction_type_ol_single_tetraplex_2v2(icmplx, cmplx_key)
        
        else:
            raise ValueError('unexpected key for tetraplex')
    

    def identify_reaction_types_ol_all_complexes(self):

        self.react_type_ol_2_index = {}
        index = 0
        for cmplx_key in self.key2indexcmplx.keys():
            ls = cmplx_key[0]
            if not ls in self.react_type_ol_2_index:
                self.react_type_ol_2_index[ls] = index
                index += 1
        self.index_2_react_type_ol = {value:key for key, value \
                                      in self.react_type_ol_2_index.items()}
        self.react_type_ol_2_cmplx_indices = \
            {index:[] for index in self.react_type_ol_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_ol_single_triplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_ol_single_tetraplex(icmplx, cmplxkey)


    def identify_reaction_type_hl_single_triplex(self, icmplx, cmplx_key):

        Ldown, Lup1, Lup2 = cmplx_key[0]
        i, j = cmplx_key[1]
        Lo1, Lo2 = self.identify_overlap_individual_triplex(i,j,Lup1,Lup2,Ldown)
        
        if (i+Lup1==j):
            rt = (min([Lo1,Lo2]),max([Lo1,Lo2]))
            rt_index = self.react_type_hl_2_index[rt]
            self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)

    
    def identify_reaction_type_hl_single_tetraplex_1v3(self, icmplx, cmplx_key):

        Ldown, Lup1, Lup2, Lup3 = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]
        Lo1, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_1v3(\
            i,j,ktop,Lup1,Lup2,Lup3,Ldown)

        if (i+Lup1==j):
            rt = (min([Lo1,Lo2]), max([Lo1,Lo2]))
            rt_index = self.react_type_hl_2_index[rt]
            self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)
        
        if (j+Lup2==ktop):
            rt = (min([Lo2,Lo3]), max([Lo2,Lo3]))
            rt_index = self.react_type_hl_2_index[rt]
            self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)

    
    def identify_reaction_type_hl_single_tetraplex_2v2(self, icmplx, cmplx_key):

        Ldown1, Lup1, Lup2, Ldown2 = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if i+Lup1 > Ldown1:
            # s-type

            if i+Lup1==j:
                _, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_2v2_stype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                rt = (min([Lo2,Lo3]), max([Lo2,Lo3]))
                rt_index = self.react_type_hl_2_index[rt]
                self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)

            if Ldown1==kbot:
                Lo1, Lo2, _ = self.identify_overlap_individual_tetraplex_2v2_stype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                rt = (min([Lo1,Lo2]), max([Lo1,Lo2]))
                rt_index = self.react_type_hl_2_index[rt]
                self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)
        
        elif i+Lup1 < Ldown1:
            # z-type

            if i+Lup1==j:
                Lo1, Lo2, _ = self.identify_overlap_individual_tetraplex_2v2_ztype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                rt = (min([Lo1,Lo2]), max([Lo1,Lo2]))
                rt_index = self.react_type_hl_2_index[rt]
                self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)
            
            if Ldown1==kbot:
                _, Lo2, Lo3 = self.identify_overlap_individual_tetraplex_2v2_ztype(\
                    i, j, kbot, Lup1, Lup2, Ldown1, Ldown2)
                rt = (min([Lo2,Lo3]), max([Lo2,Lo3]))
                rt_index = self.react_type_hl_2_index[rt]
                self.react_type_hl_2_cmplx_indices[rt_index].append(icmplx)
        

    def identify_reaction_type_hl_single_tetraplex(self, icmplx, cmplx_key):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_reaction_type_hl_single_tetraplex_1v3(icmplx, cmplx_key)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_reaction_type_hl_single_tetraplex_2v2(icmplx, cmplx_key)
        
        else:
            raise ValueError('unexpected key for tetraplex')


    def identify_reaction_types_hl_all_complexes(self):

        self.react_type_hl_2_index = {}
        index = 0
        for o1 in range(1, max(self.Ls)):
            for o2 in range(o1, max(self.Ls)):
                self.react_type_hl_2_index[(o1,o2)] = index
                index += 1
        self.index_2_react_type_hl = {value:key for key, value \
                                      in self.react_type_hl_2_index.items()}
        self.react_type_hl_2_cmplx_indices = \
            {index:[] for index in self.react_type_hl_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_hl_single_triplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_hl_single_tetraplex(icmplx, cmplxkey)


    def identify_reaction_type_an_single_triplex(self, icmplx, cmplx_key):

        _, Lup1, Lup2 = cmplx_key[0]
        i, j = cmplx_key[1]
        
        if (i+Lup1==j):
            rt = min([Lup1,Lup2])
            rt_index = self.react_type_an_2_index[rt]
            self.react_type_an_2_cmplx_indices[rt_index].append(icmplx)


    def identify_reaction_type_an_single_tetraplex_1v3(self, icmplx, cmplx_key):

        _, Lup1, Lup2, Lup3 = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]
        
        if (i+Lup1==j):
            rt = min([Lup1,Lup2])
            rt_index = self.react_type_an_2_index[rt]
            self.react_type_an_2_cmplx_indices[rt_index].append(icmplx)
        
        if (j+Lup2==ktop):
            rt = min([Lup2,Lup3])
            rt_index = self.react_type_an_2_index[rt]
            self.react_type_an_2_cmplx_indices[rt_index].append(icmplx)
        
    
    def identify_reaction_type_an_single_tetraplex_2v2(self, icmplx, cmplx_key):

        Ldown1, Lup1, Lup2, Ldown2 = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if i+Lup1==j:
            rt = min([Lup1,Lup2])
            rt_index = self.react_type_an_2_index[rt]
            self.react_type_an_2_cmplx_indices[rt_index].append(icmplx)

        if Ldown1==kbot:
            rt = min([Ldown1,Ldown2])
            rt_index = self.react_type_an_2_index[rt]
            self.react_type_an_2_cmplx_indices[rt_index].append(icmplx)
    

    def identify_reaction_type_an_single_tetraplex(self, icmplx, cmplx_key):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_reaction_type_an_single_tetraplex_1v3(icmplx, cmplx_key)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_reaction_type_an_single_tetraplex_2v2(icmplx, cmplx_key)
        
        else:
            raise ValueError('unexpected key for tetraplex')


    def identify_reaction_types_an_all_complexes(self):

        self.react_type_an_2_index = {L:i for i, L in enumerate(self.Ls)}
        self.index_2_react_type_an = \
            {value:key for key, value in self.react_type_an_2_index.items()}
        self.react_type_an_2_cmplx_indices = \
            {index:[] for index in self.react_type_an_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 3: # triplex
                self.identify_reaction_type_an_single_triplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_reaction_type_an_single_tetraplex(icmplx, cmplxkey)
        
    
    def identify_inert_type_single_duplex(self, icmplx, cmplx_key):
        it_index = self.inert_type_2_index[cmplx_key[0]]
        self.inert_type_2_cmplx_indices[it_index].append(icmplx)

    
    def identify_inert_type_single_triplex(self, icmplx, cmplx_key):
        
        _, Lup1, _ = cmplx_key[0]
        i, j = cmplx_key[1]

        if i+Lup1 != j:
            it_index = self.inert_type_2_index[cmplx_key[0]]
            self.inert_type_2_cmplx_indices[it_index].append(icmplx)

    
    def identify_inert_type_single_tetraplex_1v3(self, icmplx, cmplx_key):

        _, Lup1, Lup2, _ = cmplx_key[0]
        i, j, ktop, _ = cmplx_key[1]

        if (i+Lup1 != j) and (j+Lup2 != ktop):
            it_index = self.inert_type_2_index[cmplx_key[0]]
            self.inert_type_2_cmplx_indices[it_index].append(icmplx)
    

    def identify_inert_type_single_tetraplex_2v2(self, icmplx, cmplx_key):

        Ldown1, Lup1, _, _ = cmplx_key[0]
        i, j, _, kbot = cmplx_key[1]

        if (i+Lup1 != j) and (Ldown1 != kbot):
            it_index = self.inert_type_2_index[cmplx_key[0]]
            self.inert_type_2_cmplx_indices[it_index].append(icmplx)
    

    def identify_inert_type_single_tetraplex(self, icmplx, cmplx_key):

        if (not cmplx_key[1][2] is None) and (cmplx_key[1][3] is None):
            # type 1v3
            self.identify_inert_type_single_tetraplex_1v3(icmplx, cmplx_key)
        
        elif (cmplx_key[1][2] is None) and (not cmplx_key[1][3] is None):
            # type 2v2
            self.identify_inert_type_single_tetraplex_2v2(icmplx, cmplx_key)
    

    def identify_inert_types_all_complexes(self):

        self.inert_type_2_index = {}
        index = 0
        for cmplx_key in self.key2indexcmplx:
            ls = cmplx_key[0]
            if not ls in self.inert_type_2_index:
                self.inert_type_2_index[ls] = index
                index += 1
        self.index_2_inert_type = {value:key for key, value \
                                   in self.inert_type_2_index.items()}
        self.inert_type_2_cmplx_indices = \
            {index:[] for index in self.inert_type_2_index.values()}
        
        for cmplxkey, icmplx in self.key2indexcmplx.items():
            if len(cmplxkey[0]) == 2: # duplex
                self.identify_inert_type_single_duplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 3: # triplex
                self.identify_inert_type_single_triplex(icmplx, cmplxkey)
            elif len(cmplxkey[0]) == 4: # tetraplex
                self.identify_inert_type_single_tetraplex(icmplx, cmplxkey)
    

    def convert_energies_to_array(self):
        self.energies = np.asarray(self.energies)
    

    def convert_combinatoric_factors_to_array(self):
        self.combs = np.asarray(self.combs)
        self.combs_log = np.log(self.combs)


    def compute_binding_constants(self):
        self.Kds = np.exp(self.energies)
        self.Kds_log = np.log(self.Kds)
