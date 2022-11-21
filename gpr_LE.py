import torch
import numpy as np
import scipy as sp
from scipy import optimize
from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from spherical_bessel_zeros import Jn_zeros
from scipy.integrate import quadrature

from equistore import Labels, TensorBlock, TensorMap
from rascaline import SoapPowerSpectrum

from dataset_processing import get_dataset_slices
from error_measures import get_sse, get_rmse, get_mae, get_sae
from validation import ValidationCycle

from LE_ps import get_LE_ps

import radial_transforms

import tqdm

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
TARGET_KEY = sys.argv[5]
n_train = int(sys.argv[6])
n_test = int(sys.argv[7])
rad_tr_displacement = float(sys.argv[8])
###########################################
###########################################

date_time = datetime.now()
date_time = date_time.strftime("%m-%d-%Y-%H-%M-%S-%f")
spline_path = "splines/splines-" + date_time + ".txt"

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5
EV_TO_KCALMOL = HARTREE_TO_KCALMOL/HARTREE_TO_EV
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

n_validation_splits = 10
assert n_train % n_validation_splits == 0
n_validation = n_train // n_validation_splits
n_train_sub = n_train - n_validation

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

#a = 4.5  # Radius of the sphere
E_max_2 = 400 # try 250 (lower resolution)

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
#     factor = 2.0
#     x = a*(1-np.exp(-factor*np.tan(np.pi*r/(2*a))))
#     return x

def select_radial_transform(r, factor, a, rad_tr_dis):
    if rad_tr_selection == 1:
        radial_transform = radial_transforms.radial_transform_1(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 2:
        radial_transform = radial_transforms.radial_transform_2(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 3:
        radial_transform = radial_transforms.radial_transform_3(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 4:
        radial_transform = radial_transforms.radial_transform_4(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 5:
        radial_transform = radial_transforms.radial_transform_5(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 6:
        radial_transform = radial_transforms.radial_transform_6(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 7:
        radial_transform = radial_transforms.radial_transform_7(r, factor, a, rad_tr_dis)
    elif rad_tr_selection == 8:
        radial_transform = radial_transforms.radial_transform_8(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 9:
        radial_transform = radial_transforms.radial_transform_9(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 10:
        radial_transform = radial_transforms.radial_transform_10(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 11:
        radial_transform = radial_transforms.radial_transform_11(r, factor, a, rad_tr_dis) 
    elif rad_tr_selection == 12:
        radial_transform = radial_transforms.radial_transform_12(r, factor, a, rad_tr_dis)     
    # normalized versions below, names appended by 000
    elif rad_tr_selection == 2000:
        radial_transform = radial_transforms.radial_transform_2000(r, factor, a)
    elif rad_tr_selection == 3000:
        radial_transform = radial_transforms.radial_transform_3000(r, factor, a)
    elif rad_tr_selection == 4000:
        radial_transform = radial_transforms.radial_transform_4000(r, factor, a)
    elif rad_tr_selection == 7000:
        radial_transform = radial_transforms.radial_transform_7000(r, factor, a)
    elif rad_tr_selection == 8000:
        radial_transform = radial_transforms.radial_transform_8000(r, factor, a)
    else:
        print('NO MATCHING RADIAL TRANSFORM FOUND')
    return radial_transform

def get_LE_radial_transform(n, l, r):
    # Calculates radially transformed LE radial basis function for a 1D array of values r.
    x = select_radial_transform(r, rad_tr_factor, a, rad_tr_displacement)
    return get_LE_function(n, l, x)

# Feed LE (delta) radial spline points to Rust calculator:

n_spline_points = 101
spline_x = np.linspace(0.0, a, n_spline_points)  # x values

def function_for_splining(n, l, x):
    return get_LE_radial_transform(n, l, x)

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

print("Calculating power spectrum", flush = True)
train_ps = get_LE_ps(train_structures, spline_path, E_nl, E_max_2, a)
test_ps = get_LE_ps(test_structures, spline_path, E_nl, E_max_2, a)

all_species = np.unique(np.concatenate([train_ps.keys["a_i"], test_ps.keys["a_i"]]))

print("Expansion coefficients done", flush = True)

'''
# Normalization (???)
L2_mean = get_L2_mean(train_coefs)
#print(L2_mean)
for key in train_coefs.keys():
    train_coefs[key] /= np.sqrt(L2_mean)
    test_coefs[key] /= np.sqrt(L2_mean)
'''

# Kernel computation

def compute_kernel(first, second):
    all_species = np.unique(np.concatenate([first.keys["a_i"], second.keys["a_i"]]))

    n_first = len(np.unique(
        np.concatenate(
            [first.block(a_i=center_species).samples["structure"] for center_species in np.unique(first.keys["a_i"])]
            )))
    n_second = len(np.unique(
        np.concatenate(
            [second.block(a_i=center_species).samples["structure"] for center_species in second.keys["a_i"]]
            )))
    
    structure_kernel = torch.zeros((n_first, n_second))
  
    for center_species in all_species:
        # if center_species == 1: continue  # UNCOMMENT FOR METHANE DATASET C-ONLY VERSION
        print(f"     Calculating kernels for center species {center_species}", flush = True)
        try:
            structures_first = first.block(a_i=center_species).samples["structure"]
        except ValueError:
            print("First does not contain the above center species")
            continue
        try:
            structures_second = second.block(a_i=center_species).samples["structure"]
        except ValueError:
            print("Second does not contain the above center species")
            continue
        len_first = structures_first.shape[0]
        len_second = structures_second.shape[0]

        center_kernel = first.block(a_i=center_species).values @ second.block(a_i=center_species).values.T
        center_kernel = center_kernel**2


        for i_1 in tqdm.tqdm(range(len_first)): ## Added this for debug mode
            for i_2 in range(len_second):
                structure_kernel[structures_first[i_1], structures_second[i_2]] += center_kernel[i_1, i_2]
        # for i_1 in range(len_first):
        #     for i_2 in range(len_second):
        #         structure_kernel[structures_first[i_1], structures_second[i_2]] += center_kernel[i_1, i_2]

    return structure_kernel

print("Computing training kernel")
train_train_kernel = compute_kernel(train_ps, train_ps)
print("Training kernel done")
print("Computing test kernel")
train_test_kernel = compute_kernel(train_ps, test_ps)
print("Test kernel done")

train_train_kernel = train_train_kernel.data.cpu()
train_test_kernel = train_test_kernel.data.cpu()

print("Calculating composition features", flush = True)
X_train = get_composition_features(train_structures, all_species)
X_test = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR


# nu = 0 contribution

if "methane" in DATASET_PATH or "ch4" in DATASET_PATH:
    mean_train_energy = torch.mean(train_energies)
    train_energies -= mean_train_energy
    test_energies -= mean_train_energy
else:
    c_comp = torch.linalg.solve(X_train.T @ X_train, X_train.T @ train_energies)
    train_energies -= X_train @ c_comp
    test_energies -= X_test @ c_comp

# Validation to optimize kernel regularization
target_list = []
alpha_exp_list = np.linspace(-8, 4, 61)
for alpha_exp in alpha_exp_list:    

    validation_loss = 0.0
    for i_validation_split in range(n_validation_splits):
        index_validation_start = i_validation_split*n_validation
        index_validation_stop = index_validation_start + n_validation

        K_train_sub = torch.empty((n_train_sub, n_train_sub))
        K_train_sub[:index_validation_start, :index_validation_start] = train_train_kernel[:index_validation_start, :index_validation_start]
        if i_validation_split != n_validation_splits - 1:
            K_train_sub[:index_validation_start, index_validation_start:] = train_train_kernel[:index_validation_start, index_validation_stop:]
            K_train_sub[index_validation_start:, :index_validation_start] = train_train_kernel[index_validation_stop:, :index_validation_start]
            K_train_sub[index_validation_start:, index_validation_start:] = train_train_kernel[index_validation_stop:, index_validation_stop:]
        y_train_sub = train_energies[:index_validation_start]
        if i_validation_split != n_validation_splits - 1:
            y_train_sub = torch.concat([y_train_sub, train_energies[index_validation_stop:]])

        K_validation = train_train_kernel[index_validation_start:index_validation_stop, :index_validation_start]
        if i_validation_split != n_validation_splits - 1:
            K_validation = torch.concat([K_validation, train_train_kernel[index_validation_start:index_validation_stop, index_validation_stop:]], dim = 1)
        y_validation = train_energies[index_validation_start:index_validation_stop] 

        try:
            c_comp = torch.linalg.solve(
                K_train_sub +
                10.0**alpha_exp * torch.eye(n_train_sub), 
                y_train_sub
            )
        except:
            validation_loss += 1e30
            continue

        validation_predictions = K_validation @ c_comp

        if "qm9" in DATASET_PATH:  # optimize on the MAE for QM9
            validation_loss += get_sae(validation_predictions, y_validation).item()
        else:
            validation_loss += get_sse(validation_predictions, y_validation).item()

    if "qm9" in DATASET_PATH:
        validation_loss = validation_loss/n_train
    else:
        validation_loss = np.sqrt(validation_loss/n_train)
    print(alpha_exp, validation_loss)
    target_list.append(validation_loss)

best_alpha = alpha_exp_list[np.argmin(target_list)]
print("Result sigma optimization: ", best_alpha, min(target_list))

c = torch.linalg.solve(
    train_train_kernel +  # nu = 1, ..., 4 kernels
    10**best_alpha * torch.eye(n_train)  # regularization
    , 
    train_energies
)

train_predictions = train_train_kernel.T @ c
test_predictions = train_test_kernel.T @ c



#HYPERPARAMS
print('Cutoff Radius = ', a)
print('Selected Radial Transform = ', rad_tr_selection)
print('factor = ', rad_tr_factor)
print('displacement = ', rad_tr_displacement)
print('dataset = ', DATASET_PATH)
print('n_train = ', n_train)
print('n_test = ', n_test)

#TRAIN & TEST RMSE
print(f"Train RMSE: {get_rmse(train_predictions, train_energies).item()} [MAE: {get_mae(train_predictions, train_energies).item()}]")
print(f"Test RMSE: {get_rmse(test_predictions, test_energies).item()} [MAE: {get_mae(test_predictions, test_energies).item()}]")
'''
# Version for gradient-based local optimization
c = torch.linalg.solve(
    train_train_kernel @ best_coefficients.squeeze(dim = 0) +  # nu = 1, ..., 4 kernels
    best_sigma * torch.eye(n_train)  # regularization
    , 
    train_energies)
test_predictions = (train_test_kernel @ best_coefficients.squeeze(dim = 0)).T @ c
print(f"Test set RMSE (after kernel mixing): {get_rmse(test_predictions, test_energies).item()}")
print()
print("Final result (test MAE):")
print(n_train, get_mae(test_predictions, test_energies).item())
'''

# Clean up the spline file:
#os.remove(spline_path) # do this in Bash script (if you cancel a job it will still get deleted eventually)