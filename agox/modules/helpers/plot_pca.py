import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
from ase.io import read
from agox.modules.databases import Database
from agox.modules.samplers.spectral_graph_sampler import SpectralGraphSampler
#from agox.modules.samplers.spectral_graph_sampler_w_kmeans import SpectralGraphSamplerKMeans
#from agox.modules.samplers.sampler_kmeans import SamplerKMeans

from agox.modules.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from agox.modules.models.gaussian_process.default_model import get_default_GPR_model
from agox.modules.models.model_GPR import ModelGPR

from agox.modules.candidates.candidate_standard import StandardCandidate
from ase.calculators.singlepoint import SinglePointCalculator

from sklearn.decomposition import PCA
from ase.calculators.singlepoint import SinglePointCalculator as SPC
import matplotlib.colors as colors

def copy_calculator_from_to(orig_atoms, atoms):
    '''                                                                                                                                         
    Copy current calculator and attach to the atoms object                                                                                      
    '''
    if orig_atoms.calc is not None and 'energy' in orig_atoms.calc.results:
        if 'forces' in orig_atoms.calc.results:
            calc = SinglePointCalculator(atoms, energy=orig_atoms.calc.results['energy'],
                                         forces=orig_atoms.calc.results['forces'])
        else:
            calc = SinglePointCalculator(atoms, energy=orig_atoms.calc.results['energy'])
        atoms.set_calculator(calc)

def convert_to_candidate_object(atoms_type_object, template):
    candidate =  StandardCandidate(template=template, positions=atoms_type_object.positions, numbers=atoms_type_object.numbers,
                                          cell=atoms_type_object.cell)
    return candidate

class NewMemory():
    def __init__(self, structures):
        self.candidates = []
        template = structures[0][:1]
        for structure in structures:
            x = convert_to_candidate_object(structure,template)
            copy_calculator_from_to(structure,x)
            self.candidates.append(x)

    def get_all_candidates(self):
        return self.candidates

def center_2d(atoms):
    cell = atoms.get_cell()

    pos = atoms.positions - cell[0]/2 - cell[1]/2
    
    atoms.set_positions(pos)


def align_2d(atoms):
    pca = PCA(n_components = 2)

    N_atoms = len(atoms)
    pos = pca.fit_transform(atoms.get_positions())
    pos = np.concatenate((pos, np.zeros(N_atoms).reshape(-1, 1)), axis = 1)

    atoms.set_positions(pos)


def plot_atoms(ax, atoms, N_color_white=None, z_height=None):


    if 23 in atoms.numbers:
        N_color_white = None
        z_height = 8.7

    acols = np.array([jmol_colors[atom.number] for atom in atoms])

    if N_color_white is not None:
        acols[:N_color_white] = [1,1,1]
    if z_height is not None:
        for i in range(len(atoms)):
            if atoms[i].position[2] < z_height:
                acols[i] = [1,1,1]
                
    # enforce purple coloring of V atoms
    for i in range(len(atoms)):
       if atoms[i].number == 23:
           acols[i] = jmol_colors[25]

    ecol = [0,0,0]

    cell = atoms.get_cell()

    for ia in range(len(atoms)):
        acol = acols[ia]
        arad = covalent_radii[atoms[ia].number]*0.9

        pos = atoms[ia].position

        for d in [[0,0,0],-cell[0],cell[0],cell[1]-cell[0],cell[1],cell[1]+cell[0],-cell[1]-cell[0],-cell[1],-cell[1]+cell[0]]:
            circ = Circle([pos[0]+d[0],pos[1]+d[1]],
                      fc=acol,
                      ec=ecol,
                      radius=arad,
                      lw=0.5,
                      zorder=1+pos[2]/1000)

            ax.add_patch(circ)

def plot_10x10(structures, filename, ncols = 10, nrows = 10, size_inch = 1.7/2, limits=[-6,6]):

    no_struc = (0,None)
    plot_data = []
    # add empty row
    plot_data.append([])
    for _ in range(ncols+2):
        plot_data[0].append(no_struc)
    # add real rows
    for jdx, structs in enumerate(structures):
        plot_data.append([])
        # add empty col
        plot_data[-1].append(no_struc)
        for idx, strucs in enumerate(structs):
            if strucs == []:
                plot_data[-1].append(no_struc)
                continue
            sorted_strucs = sorted(strucs, key=lambda x: x.get_potential_energy())
            struc = sorted_strucs[0]
            E = struc.get_potential_energy()
            plot_data[-1].append((E,struc))
        # add empty col
        plot_data[-1].append(no_struc)
    # add empty row
    plot_data.append([])
    for _ in range(ncols+2):
        plot_data[-1].append(no_struc)
            
    for i,pd in enumerate(plot_data):
        print(i,len(pd))
    
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows , figsize = (ncols * size_inch, nrows * size_inch))

    for jdx, structs in enumerate(structures):
        for idx, strucs in enumerate(structs):
            #print(idx,jdx,len(strucs))
            
            if strucs == []:
                axes[jdx][idx].set_xticks([])
                axes[jdx][idx].set_yticks([])
                continue
            axes[jdx][idx].set_xlim(limits)
            axes[jdx][idx].set_ylim(limits)
            axes[jdx][idx].set_xticks([])
            axes[jdx][idx].set_yticks([])
            
            #sorted_strucs = sorted(strucs, key=lambda x: x.get_potential_energy())
            #struc = sorted_strucs[0]
            E = plot_data[jdx+1][idx+1][0]
            E1 = plot_data[jdx][idx+1][0]
            E2 = plot_data[jdx+2][idx+1][0]
            E3 = plot_data[jdx+1][idx][0]
            E4 = plot_data[jdx+1][idx+2][0]
            print(jdx,idx,E,E1,E2,E3,E4,E < min(E1,E2,E3,E4),min(E1,E2,E3,E4))
            e_color = [1,0,0] if E < min(E1,E2,E3,E4) else 'k'
            struc = plot_data[jdx+1][idx+1][1].copy()
            center_2d(struc)
            plot_atoms(axes[jdx][idx], struc, N_color_white=None, z_height=None)
            axes[jdx][idx].set_aspect('equal')
            axes[jdx][idx].text(0.9*limits[0],0.9*limits[1],r'${{{:d}}}$'.format(len(strucs)),verticalalignment='top',bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            axes[jdx][idx].text(0.95*limits[1],0.9*limits[0],r'${{{:8.3f}}}$'.format(E),horizontalalignment='right',fontsize=8,rotation='vertical',bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'), color=e_color)
                
    fig.tight_layout()
    plt.subplots_adjust(hspace = 0, wspace = 0)
    plt.savefig(filename)
    plt.close()


def plot_as_pca(strucs,sample,new_strucs,episode,model_calculator):

    if len(strucs) < 2:
        return

    if True:
        N_color_white = None
        z_height = None
    else:
        N_color_white = 100
        z_height = 6.6

    temp_atoms = strucs[0]
    feature_calculator = Angular_Fingerprint(temp_atoms,Rc1=6,Rc2=4,binwidth1=0.2,Nbins2=30,
                                             sigma1=0.2,sigma2=0.2,gamma=2,eta=20,use_angular=True)

    # Calculate features
    f = feature_calculator.get_featureMat(strucs)

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(f)

    # Project data to 2D with PCA
    f2d = pca.transform(f)

    xmax = max(f2d[:,0])
    xmin = min(f2d[:,0])
    xspan = xmax - xmin
    xbins = np.linspace(xmin,xmax + 1e-8 * xspan,11)[1:]

    ymax = max(f2d[:,1])
    ymin = min(f2d[:,1])
    yspan = ymax - ymin
    ybins = np.linspace(ymin,ymax + 1e-8 * yspan,11)[1:]

    print('_PCA1_: {:8.3f}'.format(xspan),xmin,xmax)
    print('_PCA2_: {:8.3f}'.format(yspan),ymin,ymax)


    all_strucs = []
    all_strucs.append(strucs)
    all_strucs.append(sample)
    all_strucs.append(new_strucs)
    dnum = 3
    if list(set(all_strucs[0][0].get_atomic_numbers())) == [1, 6, 7]:
        all_strucs.append(read('/home/hammer/agox2021gitmaster/runs/pyridine/analysis/best_twelwe_pyridine_isomers.traj',index=':'))
        dnum += 1
        all_strucs.append([])
        dnum += 1

        for candidate in all_strucs[-2]:
            atoms = StandardCandidate(template=all_strucs[0][0].template, positions=candidate.positions, numbers=candidate.numbers, cell=candidate.cell)
    
            if model_calculator.ready_state:
                atoms.set_calculator(model_calculator)
                e = atoms.get_potential_energy()
            else:
                e = 0
            atoms.set_calculator(SPC(atoms, energy=e))
            all_strucs[-1].append(atoms)
            
    for index_one_restart in range(dnum):
        if len(all_strucs[index_one_restart]) < 2:
            continue

        structure_for_map = []
        for i in range(10):
            structure_for_map.append([])
            for j in range(10):
                structure_for_map[-1].append([])

        # Calculate features
        g = feature_calculator.get_featureMat(all_strucs[index_one_restart])

        # Project data to 2D with PCA
        g2d = pca.transform(g)
    
        xbin_indices = np.digitize(g2d[:,0],xbins)
        ybin_indices = np.digitize(g2d[:,1],ybins)

        structures_for_sample = []
        for j in range(len(ybins)):
            for i in range(len(xbins)):
                relevant_ones = (xbin_indices==i) & (ybin_indices==j)
                ks = np.arange(len(all_strucs[index_one_restart]))[relevant_ones]
                for k in ks:
                    structure_for_map[j][i].append(all_strucs[index_one_restart][k])

        plot_10x10(structure_for_map,'pca_strucs_{:05d}_{}.png'.format(episode,index_one_restart))
