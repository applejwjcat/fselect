import h5py as h5
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import scipy.optimize as so
from itertools import product
from scipy import fftpack as fp


def gather_data1(f, iv="velx", verbose=0):
    value = f[iv][()]
    block_size = f["block size"]
    size = (1 / np.mean(block_size, axis=0)).astype(np.int8)
    shape = np.array(value.shape[-3:])
    field = np.empty(shape * size)
    for k in range(size[0]):
        for j in range(size[1]):
            for i in range(size[2]):
                field[shape[0] * i:shape[0] * (i + 1),
                      shape[1] * j:shape[1] * (j + 1), shape[2] * k:shape[2] *
                      (k + 1)] = value[k * size[2] ** 2 + j * size[1] + i]
    return field


def gather_data2(f, iv="velx", verbose=0):
    value = f[iv][()]
    block_size = f["block size"]
    size = (1 / np.mean(block_size, axis=0)).astype(np.int8)
    # shape=value.shape[-3:]
    n = size[0]
    gather_x = [
        np.concatenate([value[k * n * n + j * n + i] for i in range(size[0])],
                       axis=0) for k in range(size[2]) for j in range(size[1])
    ]
    gather_y = [
        np.concatenate([gather_x[k * n + j] for j in range(size[1])], axis=1)
        for k in range(size[2])
    ]
    return np.concatenate([gather_y[k] for k in range(size[2])], axis=2)


def gather_data3(f, iv="velx", verbose=0):
    value = f[iv][()]
    block_size = f["block size"]
    size = (1 / np.mean(block_size, axis=0)).astype(np.int8)
    shape = np.array(value.shape[-3:])
    field = np.empty(shape * size)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                field[shape[0] * i:shape[0] * (i + 1),
                      shape[1] * j:shape[1] * (j + 1), shape[2] * k:shape[2] *
                      (k + 1)] = value[i * size[2] ** 2 + j * size[1] + k]
    return field


def gather_data4(f, iv="velx", verbose=0):
    value = f[iv][()]
    block_size = f["block size"]
    size = (1 / np.mean(block_size, axis=0)).astype(np.int8)
    # shape=value.shape[-3:]
    n = size[0]
    gather_x = [
        np.concatenate([value[i * n * n + j * n + k] for k in range(size[2])],
                       axis=2) for i in range(size[0]) for j in range(size[1])
    ]
    gather_y = [
        np.concatenate([gather_x[i * n + j] for j in range(size[1])], axis=1)
        for i in range(size[0])
    ]
    return np.concatenate([gather_y[i] for i in range(size[0])], axis=0)


def fft_energy(file):
    f = h5.File(file, 'r')

    fieldx = gather_data1(f, iv='velx')
    fieldy = gather_data1(f, iv='vely')
    fieldz = gather_data1(f, iv='velz')

    nx, ny, nz = fieldx.shape
    # x1 = (np.arange(nx) - nx // 2).reshape(nx, 1, 1)
    # x2 = (np.arange(ny) - ny // 2).reshape(1, ny, 1)
    # x3 = (np.arange(nz) - nz // 2).reshape(1, 1, nz)
    # rad = np.sqrt(np.square(x1) + np.square(x2) + np.square(x3))

    kx_field = fp.fftn(fieldx) / nx ** 3
    ky_field = fp.fftn(fieldy) / ny ** 3
    kz_field = fp.fftn(fieldz) / nz ** 3

    kx_field = np.roll(kx_field, (nx//2, ny//2, nz//2), axis=(0, 1, 2))
    ky_field = np.roll(ky_field, (nx//2, ny//2, nz//2), axis=(0, 1, 2))
    kz_field = np.roll(kz_field, (nx//2, ny//2, nz//2), axis=(0, 1, 2))

    # rad_unique = np.unique(np.array(rad.flatten()))
    # length = len(rad_unique)

    return 0.5*(np.square(np.abs(kx_field))+np.square(np.abs(ky_field))+np.square(np.abs(kz_field)))


def index_loc(ndim):
    ndims = ndim
    i_lst = range(-ndims//2, ndims//2)
    indices = np.array(list(product(i_lst, i_lst, i_lst)))
    M = (indices**2).sum(axis=1)

    df = pd.DataFrame(indices, columns=['i', 'j', 'k'])
    df['M'] = M
    df = df.set_index('M')
    df = df.sort_index()
    df += ndims//2
    return (M, df)


def gather_energy(M, df, E_field):
    energy_spectrum = {}
    # for m in tqdm(set(M)):
    for m in set(M):
        energy_spectrum[m] = E_field[tuple(df.loc[m].values.T)].sum()
    # energy_df = pd.DataFrame(pd.Series(energy_spectrum), columns=['energy'])
    # energy_df = energy_df.reset_index().rename(columns={'index': 'wave_num'})
    # energy_df['wave_num'] = 2*np.pi*np.sqrt(energy_df['wave_num'])
    energy_df = pd.DataFrame.from_dict(
        energy_spectrum, orient="index", columns=['energy'])
    energy_df.index = 2*np.pi*np.sqrt(energy_df.index)
    return energy_df


def analysis_energy(energy_df):
    imax = int(energy_df.wave_num.max()/np.pi)+1
    # imax
    bins = np.arange(1, imax, 2)*np.pi
    bins = np.insert(bins, 0, 0)
    energy_df["group"] = pd.cut(energy_df["wave_num"], bins=bins, labels=range(
        len(bins)-1), right=False, include_lowest=True)
    # energy_df
    # bins
    bin_wave = energy_df["wave_num"].groupby(energy_df["group"]).mean()
    bin_energy = energy_df["energy"].groupby(energy_df["group"]).sum()
    return (bin_wave, bin_energy)


# def show_spectrum(energy_df):
    # plt.loglog(energy_df['wave_num'], energy_df['energy'])
    # plt.savefig('xxx.png')
