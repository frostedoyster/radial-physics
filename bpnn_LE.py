import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import scipy as sp
from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from spherical_bessel_zeros import Jn_zeros
from scipy.integrate import quadrature

import rascaline
rascaline._c_lib._get_library()

from equistore import Labels, TensorBlock, TensorMap
from dataset_processing import get_dataset_slices
from dataset_bpnn import AtomisticDataset, create_dataloader

import tqdm
import error_measures
import radial_transforms

from datetime import datetime
import os

###########################################
###########################################
### HERE WE DEFINE THE INPUTS PASSED AS ARGUMENTS FROM OUR BASH SCRIPT (and convert them to floats/int, if necessary)
import sys
main_name = sys.argv[0]
a = float(sys.argv[1])
rad_tr_selection = float(sys.argv[2])
rad_tr_factor = float(sys.argv[3])
DATASET_PATH = sys.argv[4]
n_train = int(sys.argv[5])
n_test = int(sys.argv[5])
E_max_2 = int(sys.argv[6])
rad_tr_displacement = float(sys.argv[7])
if DATASET_PATH == 'datasets/qm9.xyz':
    TARGET_KEY = 'U0'
elif DATASET_PATH == 'datasets/random-ch4-10k.extxyz':
    TARGET_KEY = 'energy'
elif DATASET_PATH == 'datasets/gold.xyz':
    TARGET_KEY = 'elec._Free_Energy_[eV]'
else:
    print("Dataset not found")
###########################################
###########################################

date_time = datetime.now()
date_time = date_time.strftime("%m-%d-%Y-%H-%M-%S-%f")
spline_path = "splines/splines-" + date_time + ".txt"

torch.set_default_dtype(torch.float64)
torch.set_num_threads(70)
print("CUDA is available: ", torch.cuda.is_available())  # See if we can use a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5
EV_TO_KCALMOL = HARTREE_TO_KCALMOL/HARTREE_TO_EV

DATASET_PATH = 'datasets/random-ch4-10k.extxyz'
TARGET_KEY = "energy" # "elec. Free Energy [eV]" # "U0"
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

# n_test = 1000
# n_train = 1000

test_slice = str(0) + ":" + str(n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train)

# Spherical expansion and composition

def get_composition_features(frames, all_species):
    species_dict = {s: i for i, s in enumerate(all_species)}
    data = torch.zeros((len(frames), len(species_dict)))
    for i, f in enumerate(frames):
        for s in f.numbers:
            data[i, species_dict[s]] += 1
    properties = Labels(
        names=["atomic_number"],
        values=np.array(list(species_dict.keys()), dtype=np.int32).reshape(
            -1, 1
        ),
    )

    frames_i = np.arange(len(frames), dtype=np.int32).reshape(-1, 1)
    samples = Labels(names=["structure"], values=frames_i)

    block = TensorBlock(
        values=data, samples=samples, components=[], properties=properties
    )
    composition = TensorMap(Labels.single(), blocks=[block])
    return composition.block().values

# a = 4.5  # Radius of the sphere
# E_max_2 = 600.0

l_big = 26
n_big = 26

z_ln = Jn_zeros(l_big, n_big)  # Spherical Bessel zeros
z_nl = z_ln.T

E_nl = z_nl**2
E_max = E_max_2 - E_nl[0, 0]
n_max = np.where(E_nl[:, 0] <= E_max)[0][-1] + 1
l_max = np.where(E_nl[0, :] <= E_max)[0][-1]
print(n_max, l_max)

def R_nl(n, l, r):
    return j_l(l, z_nl[n, l]*r/a)

def N_nl(n, l):
    # Normalization factor for LE basis functions
    def function_to_integrate_to_get_normalization_factor(x):
        return j_l(l, x)**2 * x**2
    integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l])
    return (1.0/z_nl[n, l]**3 * integral)**(-0.5)

def get_LE_function(n, l, r):
    R = np.zeros_like(r)
    for i in range(r.shape[0]):
        R[i] = R_nl(n, l, r[i])
    return N_nl(n, l)*R*a**(-1.5)

# # Radial transform
# def radial_transform(r):
#     # Function that defines the radial transform x = xi(r).
#     factor = 1.4
#     x = a*(1-np.exp(-factor*np.tan(np.pi*r/(2*a))))
#     return x

def get_LE_radial_transform(n, l, r, rad_tr_selection):
    # Calculates radially transformed LE radial basis function for a 1D array of values r.
    x = radial_transforms.select_radial_transform(r, rad_tr_factor, a, rad_tr_displacement, rad_tr_selection)
    return get_LE_function(n, l, x)

# Feed LE (delta) radial spline points to Rust calculator:

n_spline_points = 101
spline_x = np.linspace(0.0, a, n_spline_points)  # x values

def function_for_splining(n, l, x):
    return get_LE_radial_transform(n, l, x, rad_tr_selection)

spline_f = []
for l in range(l_max+1):
    for n in range(n_max):
        spline_f_single = function_for_splining(n, l, spline_x)
        spline_f.append(spline_f_single)
spline_f = np.array(spline_f).T
spline_f = spline_f.reshape(n_spline_points, l_max+1, n_max)  # f(x) values

def function_for_splining_derivative(n, l, r):
    delta = 1e-6
    all_derivatives_except_first_and_last = (function_for_splining(n, l, r[1:-1]+delta) - function_for_splining(n, l, r[1:-1]-delta)) / (2.0*delta)
    derivative_at_zero = (function_for_splining(n, l, np.array([delta/10.0])) - function_for_splining(n, l, np.array([0.0]))) / (delta/10.0)
    derivative_last = (function_for_splining(n, l, np.array([a])) - function_for_splining(n, l, np.array([a-delta/10.0]))) / (delta/10.0)
    return np.concatenate([derivative_at_zero, all_derivatives_except_first_and_last, derivative_last])

spline_df = []
for l in range(l_max+1):
    for n in range(n_max):
        spline_df_single = function_for_splining_derivative(n, l, spline_x)
        spline_df.append(spline_df_single)
spline_df = np.array(spline_df).T
spline_df = spline_df.reshape(n_spline_points, l_max+1, n_max)  # df/dx values

with open(spline_path, "w") as file:
    np.savetxt(file, spline_x.flatten(), newline=" ")
    file.write("\n")

with open(spline_path, "a") as file:
    np.savetxt(file, (1.0/(4.0*np.pi))*spline_f.flatten(), newline=" ")
    file.write("\n")
    np.savetxt(file, (1.0/(4.0*np.pi))*spline_df.flatten(), newline=" ")
    file.write("\n")

train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

train_structure_species = np.concatenate([train_structure.get_atomic_numbers() for train_structure in train_structures])
test_structure_species = np.concatenate([test_structure.get_atomic_numbers() for test_structure in test_structures])
all_species = np.unique(np.concatenate([train_structure_species, test_structure_species]))

print("Calculating composition features", flush = True)
train_comp = get_composition_features(train_structures, all_species)
test_comp = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

print("Calculating composition features", flush = True)
X_train = get_composition_features(train_structures, all_species)
X_test = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

# nu = 0 contribution
if "methane" in DATASET_PATH or "ch4" in DATASET_PATH:
    mean_train_energy = torch.mean(train_energies)
    train_energies -= mean_train_energy
    test_energies -= mean_train_energy
else:
    c_comp = torch.linalg.solve(X_train.T @ X_train, X_train.T @ train_energies)
    train_energies -= X_train @ c_comp
    test_energies -= X_test @ c_comp

# Need normalization!!!!!!!!!

hypers_spherical_expansion = {
        "cutoff": a,
        "max_radial": int(n_max),
        "max_angular": int(l_max),
        "center_atom_weight": 1.0,
        "radial_basis": {"Tabulated": {"file": spline_path}},
        "atomic_gaussian_width": 100.0,
        "cutoff_function": {"Step": {}},
    }
  
print("Creating datasets and dataloaders")  
train_dataset = AtomisticDataset(
    train_structures,
    all_species,
    spline_path, 
    E_nl, 
    E_max_2, 
    a,
    energies=train_energies,
)

test_dataset = AtomisticDataset(
    test_structures,
    all_species,
    spline_path, 
    E_nl, 
    E_max_2, 
    a,
    energies=test_energies,
)

train_dataloader = create_dataloader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    device=device,
)

test_dataloader = create_dataloader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    device=device,
)

for ps, energies, indices in train_dataloader:
    nfeat = ps.block(0).values.shape[-1]
print(f"nfeat: {nfeat}")

nrepeat = 10
avgerr = 0.0
train_avgerr = 0.0
avgerr_mae = 0.0
train_avgerr_mae = 0.0
nlayers = 3
nneurons = [32, 32, 32]
assert nlayers == len(nneurons)

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.NNs = nn.ModuleDict()
        for species in all_species:  # For each element, create a NN.
            self.NNs[str(species)] = nn.ModuleList()  # Initialise a list that will contain all the layers.
            # NNs[str(species)].append(nn.Linear(nfeat, 1, dtype = torch.float32, bias = False))
            self.NNs[str(species)].append(nn.Linear(nfeat, nneurons[0]))  # Create transformation between input layer and first hidden layer.
            for j in range(nlayers-1):
                self.NNs[str(species)].append(nn.Linear(nneurons[j], nneurons[j+1]))  # Create transformations between hidden layers..
            self.NNs[str(species)].append(nn.Linear(nneurons[nlayers-1], 1))  # Create tranformation between last hidden layer and output layer.

    def forward(self, tensor_map, original_indices):

        y_structure = torch.zeros(size = (len(original_indices), 1), device = tensor_map.block(0).values.device)

        for species in all_species:
            NN_for_current_center_species = self.NNs[str(species)]
            try:
                block = tensor_map.block(a_i=species)
            except ValueError as e:
                # print(f"Batch doesn't have element {species}")
                continue
            x = block.values

            # ylin = self.NNs[str(species)][0](x)
            yprov = x
            for k in range(nlayers):
                yprov = torch.tanh(NN_for_current_center_species[k](yprov))  # +1
            yprov = NN_for_current_center_species[nlayers](yprov)  # +1
            y = yprov  # + ylin

            indices_for_index_add = []
            for structure in block.samples["structure"]:
                indices_for_index_add.append(np.where(original_indices == structure)[0][0])
            indices_for_index_add = torch.tensor(indices_for_index_add, dtype=torch.int32)

            y_structure.index_add_(0, indices_for_index_add, y)

        return y_structure




for irepeat in range(nrepeat):       

    network = Network().to(device)
    optimizer = optim.Adam(network.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = 1e-5, eps = 1e-12, patience = 50, verbose = True)


    errmin = 10000000.0
    errmin_mae = 10000000.0
    train_errmin = 10000000.0
    train_errmin_mae = 10000000.0
    best_params = {}
    for epoch in tqdm.tqdm(range(10000)):
        initial_time = time.time()

        for ps, energies, original_indices in train_dataloader:
            optimizer.zero_grad()  # Avoid accumulation of gradients

            predictions = network(ps, original_indices)
            # print(predictions.requires_grad)
            loss = F.mse_loss(predictions, energies, reduction='sum')
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            with torch.no_grad():
                train_loss = 0.0
                train_mae = 0.0
                for ps, energies, original_indices in train_dataloader:
                    predictions = network(ps, original_indices)
                    #print(predictions.requires_grad)
                    train_loss += F.mse_loss(predictions, energies, reduction='sum').item()
                    train_mae+= error_measures.get_sae(predictions, energies).item()
                train_loss = np.sqrt(train_loss/n_train)
                train_mae=train_mae/n_train
            with torch.no_grad():
                test_loss = 0.0
                test_mae = 0.0
                for ps, energies, original_indices in test_dataloader:
                    predictions = network(ps, original_indices)
                    # print(predictions.requires_grad)
                    test_loss += F.mse_loss(predictions, energies, reduction='sum').item()
                    test_mae += error_measures.get_sae(predictions, energies).item()
                test_loss = np.sqrt(test_loss/n_test)
                test_mae = test_mae/n_test

        if (test_loss <= errmin):
            errmin = test_loss
            for name, params in network.named_parameters():
                best_params[name] = params.clone()
        
        if (test_mae <= errmin_mae):
            errmin_mae = test_mae
            for name, params in network.named_parameters():
                best_params[name] = params.clone()
    
        if (train_loss <= train_errmin):
            train_errmin = train_loss
            for name, params in network.named_parameters():
                best_params[name] = params.clone()    

        if (train_mae <= train_errmin_mae):
            train_errmin_mae = train_mae
            for name, params in network.named_parameters():
                best_params[name] = params.clone()
    
        lr = optimizer.param_groups[0]["lr"]  # This appears to be making a deep copy, for whatever reason (probably copying from GPU to CPU).
        # np.set_printoptions(precision=3)  # If this is not set the default precision will be 4 and torch.float64 numbers will look like float32 numbers.
        #print(repr((epoch+1)).rjust(6), repr(train_loss).rjust(20), repr(test_loss).rjust(20), lr, time.time()-initial_time, flush = "True")

        scheduler.step(test_loss)
        if (optimizer.param_groups[0]["lr"] <= 2e-12):
            print("Very small learning rate reached: 1e-12")
            break
        if (optimizer.param_groups[0]["lr"] != lr):   
            for name, params in network.named_parameters():
                params.data.copy_(best_params[name])
            optimizer = optim.Adam(network.parameters(), lr = lr*0.1)  # Reinitialise Adam so that it resets the moment vectors.
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = 1e-5, eps = 1e-12, patience = 50, verbose = True)  # Also reinitialise scheduler just in case.

    print(f'Train error minimum for model run no. {irepeat+1}: {train_errmin} [MAE: {train_errmin_mae}]')
    train_avgerr += train_errmin
    train_avgerr_mae += train_errmin_mae
    print(f'Test error minimum for model run no. {irepeat+1}:  {errmin} [MAE: {errmin_mae}]')
    avgerr += errmin
    avgerr_mae += errmin_mae
avgerr = avgerr/nrepeat
train_avgerr = train_avgerr/nrepeat
avgerr_mae = avgerr_mae/nrepeat
train_avgerr_mae = train_avgerr_mae/nrepeat


#HYPERPARAMS
print('E_max_2 = ', E_max_2)
print('Cutoff Radius = ', a)
print('Selected Radial Transform = ', rad_tr_selection)
print('factor = ', rad_tr_factor)
print('displacement = ', rad_tr_displacement)
print('dataset = ', DATASET_PATH)
print('n_train = ', n_train)
print('n_test = ', n_test)

#TRAIN & TEST RMSE
print(f'Train error averaged over all models Train RMSE:  {train_avgerr} [Train MAE: {train_avgerr_mae}]')
print(f'Test error averaged over all models Test RMSE: {avgerr} [Test MAE: {avgerr_mae}]')

# Clean up the spline file:
# os.remove(spline_path)