#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os.path
import pandas as pd
import re
import subprocess
import tempfile
import tqdm
# from ipdb import set_trace as st # this import has issues with multiprocesing
plt.rcParams.update({"font.family": "serif"})


# NJOY output file to use
output_unit = '20'


def parse_arguments():
    """Parse user arguments using the argparse module."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')
    # Common elements
    parent_parser = argparse.ArgumentParser(add_help=False,)
    parent_parser.add_argument(
            'pdos_file', type=argparse.FileType(), help='Path to probability '
            'density of state (PDOS) file.')
    # PDOS Plotter
    pdos_plotter_parser = subparsers.add_parser(
            'plot_pdos', parents=[parent_parser], help='Load and plot '
            'probability density of state file')
    pdos_plotter_parser.add_argument(
            'output_file', type=argparse.FileType('w'), help='Path to output '
            'file')
    # LEAPR driver
    leapr_driver_parser = subparsers.add_parser(
            'leapr', parents=[parent_parser], help='Load probability density '
            'of state and do leapr runs')
    leapr_driver_parser.add_argument(
            '--njoy-path', type=argparse.FileType(), metavar='P',
            required=True, help='Path to njoy executable')
    leapr_driver_parser.add_argument(
            '--nprocs', default=1, type=int, metavar='N', help='Number of '
            'leapr instances')
    # File 7 Parser
    parse_mf7_parser = subparsers.add_parser('parse_mf7', help='Parse ENDF '
                                             'File 7')
    parse_mf7_parser.add_argument(
            '--input-file', type=argparse.FileType(), metavar='P',
            required=True, help='Path to ENDF File 7 file')
    return parser.parse_args()


def load_pdos(args):
    """Load a probability density of state (PDOF) file."""
    return pd.read_csv(args.pdos_file, sep='  ', index_col=0, header=None,
                       engine='python')


def plot_pdos(args):
    """Plot a probability density of state (PDOS) file."""
    df = load_pdos(args)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    mean = df.mean(axis=1)
    ax.plot(df, color='black', alpha=0.1, linewidth=0.1)
    ax.plot(mean, color='black', linewidth=1.)
    # ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), color='black',
    #                 edgecolor=None, alpha=0.1)
    # x axis
    ax.set_xticks(np.linspace(df.index.min(), df.index.max(), 11))
    ax.set_xlim([df.index.min(), df.index.max()])
    ax.set_xlabel(r'Energy (meV)')
    # y axis
    ymin, ymax = 0., 0.012
    ax.set_yticks(np.linspace(ymin, ymax, 7))
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel(r'Probability Density ($\mathrm{meV}^{-1}$)')
    # save
    fig.savefig(args.output_file.buffer, format='pgf')


def parse_file7(mf7_path):
    """Parse ENDF File 7 file and return pandas DataFrame."""
    def to_float(endf_float_string):
        """Convert ENDF-style float string to float."""
        pattern = re.compile(r'([+-])')
        return float(pattern.sub(r'E\1', endf_float_string))

    with open(mf7_path, mode='r') as sab_file:
        # skip headers
        while True:
            if sab_file.readline().endswith('1 7  4    5\n'):
                N_beta = int(sab_file.readline().split()[0])
                break
        alphas, betas, Ts, Ss = [], [], [], []
        for _ in range(N_beta):
            # Read first temperature, beta block
            entries = sab_file.readline().split()
            temp, beta = (to_float(x) for x in entries[:2])
            N_temp = int(entries[2]) + 1
            # The first temperature is handled differently: alpha and
            # S(alpha, beta) values are given together.
            N_alpha = int(sab_file.readline().split()[0])
            N_full_rows, remainder = divmod(N_alpha, 3)
            for row in range(N_full_rows + (remainder != 0)):
                # Everything after column 66 is ignored
                doubles = [
                        to_float(x) for x in sab_file.readline()[:66].split()]
                for alpha, S in zip(doubles[::2], doubles[1::2]):
                    alphas.append(alpha)
                    betas.append(beta)
                    Ts.append(temp)
                    Ss.append(S)
            # The remaining temperatures are handled uniformly
            N_full_rows, remainder = divmod(N_alpha, 6)
            for _ in range(N_temp - 1):
                temp, beta = (
                        to_float(x) for x in sab_file.readline().split()[:2])
                # Subsequent betas use the first beta's alpha grid.
                unique_alphas = (a for a in alphas[:N_alpha])
                for row in range(N_full_rows + (remainder != 0)):
                    line = sab_file.readline()[:66]
                    for S in line.split():
                        alphas.append(next(unique_alphas))
                        betas.append(beta)
                        Ts.append(temp)
                        Ss.append(to_float(S))
        return pd.DataFrame.from_dict(
                {'alpha': alphas, 'beta': betas, 'T': Ts, 'S': Ss})


def run_leapr_kernel(args):
    """Runs a single instance of leapr using given PDOS values"""
    njoy_path, column_name, pdos = args
    # declare leapr run
    start = 'leapr'
    # card 1 - units
    #    nout     endf output unit for thermal file
    card1 = f'{output_unit}/'
    # card 2 - title
    card2 = "'An attempt at an evaluation'/"
    # card 3 - run control
    #    ntempr  number of temperatures (def=1)
    #    iprint  print control (0=min, 1=more, 2=most, def=1)
    #    nphon   phonon-expansion order (def=100)
    card3 = f'{len(temperatures)} 1 200/'
    # card 4 - endf output control
    #    mat     endf mat number
    #    za      1000*z+a for principal scatterer
    #    isabt   sab type (0=symmetric, 1=asymmetric, def=0)
    #    ilog    log flag (0=s, 1=log10(s), def=0)
    #    smin    minimum S(alpha, beta) stored in file (def=1e-75)
    card4 = '1 101/'
    # card 5 - principal scatterer control
    #    awr     weight ratio to neutron for principal scatterer
    #    spr     free atom cross section for principal scatterer
    #    npr     number of principal scattering atoms in compound
    card5 = '0.99917 20.478 2/'
    # card 6 - secondary scatterer control
    #    nss     number of secondary scatterers (0 or 1)
    #    b7      secondary scatterer type
    #             (0=sct only, 1=free, 2=diffusion)
    #    aws     weight ratio to neutron for secondary scatterer
    #    sps     free atoms cross section for secondary scatterer
    #    mss     number of atoms of this type in the compound
    card6 = '1 1 15.85316 3.761 1/'
    # card 7 - alpha, beta control
    #    nalpha   number of alpha values
    #    nbeta    number of beta values
    #    lat      if lat.eq.1, alpha and beta values are scaled
    #               by .0253/tev, where tev is temp in ev.  (def=0)
    card7 = '182 259 1/'
    # card 8 - alpha values (increasing order)
    card8 = ' '.join(str(a) for a in alphas) + '/'
    # card 9 - beta values (increasing order)
    card9 = ' '.join(str(b) for b in betas) + '/'
    # card 10 - temperature (k)
    card10 = f'{temperatures[0]}/'
    # card 11 -- continuous distribution control
    #    delta    interval in ev
    #    ni       number of points
    card11 = '0.0005 1001'
    # card 12 -- rho(energy) (order of increasing ev)
    card12 = ' '.join(str(rho) for rho in pdos) + '/'
    # card 13 - continuous distribution parameters
    #    twt       translational weight
    #    c         diffusion constant (zero for free gas)
    #    tbeta     normalization for continuous part
    card13 = '0. 0. 1./'  # ASK CHAPMAN ABOUT THIS
    # discrete oscillator control: number of discrete oscillators
    card14 = '0/'
    # Repeat for other temperatures
    card10extra = '/\n'.join(f'-{t}' for t in temperatures[1:]) + '/'
    # file 1 comments
    card20 = (
        "'Modeled after INDC(NDS)-0470 prepared by M. Mattes and J.        '\n"
        "'Keinert. Results by C. Chapman were used for the probability     '\n"
        "'density of states.                                               '/")
    # trailing blank line and stop
    end = '/\nstop'
    # Work inside a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # set up filenames
        input_path = os.path.join(tmpdir, f'input_{column_name}.leapr')
        output_path = os.path.join(tmpdir, f'output_{column_name}.leapr')
        mf7_path = os.path.join(tmpdir, f'tape{output_unit}')
        with open(input_path, 'w') as input_file:
            input_file.writelines('\n'.join([
                start, card1, card2, card3, card4, card5, card6, card7, card8,
                card9, card10, card11, card12, card13, card14, card10extra,
                card20, end]))
        subprocess.Popen(
                [njoy_path, '-i', input_path, '-o', output_path],
                cwd=tmpdir, stdout=subprocess.DEVNULL).wait()
        df = parse_file7(mf7_path)
        df['Realization'] = column_name
        return df


def run_leapr(args):
    """Create input files and run leapr"""
    pdos = load_pdos(args)
    # Prepare iterable for pool.starmap
    starmap_args = (
            (args.njoy_path.name, *colname_pdos_pair)
            for colname_pdos_pair in pdos.iteritems())
    dfs = []
    with mp.Pool(args.nprocs) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(
                run_leapr_kernel, starmap_args), total=len(pdos.columns)):
            dfs.append(result)
    df = pd.concat(dfs)
    df['Realization'] = df['Realization'].astype(np.uint16)
    df.to_hdf('sab.hdf5', 'sab')


alphas = (
        5.000000E-4, 1.000000E-3, 5.000000E-3, 1.000000E-2, 2.500000E-2,
        5.000000E-2, 7.500000E-2, 1.000000E-1, 1.250000E-1, 1.500000E-1,
        2.000000E-1, 2.500000E-1, 3.000000E-1, 3.250000E-1, 3.500000E-1,
        3.750000E-1, 4.000000E-1, 4.250000E-1, 4.500000E-1, 4.750000E-1,
        5.000000E-1, 5.250000E-1, 5.500000E-1, 5.800000E-1, 6.100000E-1,
        6.500000E-1, 6.900000E-1, 7.300000E-1, 7.800000E-1, 8.300000E-1,
        8.800000E-1, 9.400000E-1, 1.000000E+0, 1.080000E+0, 1.160000E+0,
        1.240000E+0, 1.330000E+0, 1.430000E+0, 1.540000E+0, 1.660000E+0,
        1.790000E+0, 1.940000E+0, 2.090000E+0, 2.260000E+0, 2.480000E+0,
        2.712700E+0, 2.890000E+0, 3.110000E+0, 3.380000E+0, 3.670000E+0,
        3.980000E+0, 4.320000E+0, 4.650000E+0, 5.000000E+0, 5.425500E+0,
        6.000000E+0, 6.560000E+0, 7.130000E+0, 7.600000E+0, 8.102600E+0,
        8.800000E+0, 9.500000E+0, 1.020000E+1, 1.081520E+1, 1.170000E+1,
        1.260000E+1, 1.352800E+1, 1.440000E+1, 1.530000E+1, 1.620510E+1,
        1.723300E+1, 1.820000E+1, 1.892000E+1, 2.030000E+1, 2.163000E+1,
        2.290000E+1, 2.430800E+1, 2.560000E+1, 2.702000E+1, 2.840000E+1,
        2.973000E+1, 3.100000E+1, 3.241000E+1, 3.344000E+1, 3.446600E+1,
        3.615000E+1, 3.718000E+1, 3.880000E+1, 4.051300E+1, 4.154000E+1,
        4.257000E+1, 4.420000E+1, 4.600000E+1, 4.700000E+1, 4.861500E+1,
        4.960000E+1, 5.120000E+1, 5.250000E+1, 5.441000E+1, 5.520000E+1,
        5.672000E+1, 5.840000E+1, 5.980000E+1, 6.120000E+1, 6.251000E+1,
        6.380000E+1, 6.523000E+1, 6.650000E+1, 6.790000E+1, 6.893000E+1,
        7.061000E+1, 7.164000E+1, 7.292000E+1, 7.590000E+1, 8.000000E+1,
        8.400000E+1, 8.900000E+1, 9.400000E+1, 1.000000E+2, 1.050000E+2,
        1.130000E+2, 1.206300E+2, 1.260000E+2, 1.320000E+2, 1.400000E+2,
        1.470000E+2, 1.540000E+2, 1.620000E+2, 1.700000E+2, 1.770000E+2,
        1.840000E+2, 1.910000E+2, 1.990000E+2, 2.080000E+2, 2.180000E+2,
        2.270000E+2, 2.370000E+2, 2.460000E+2, 2.550000E+2, 2.650000E+2,
        2.757200E+2, 2.840000E+2, 2.935800E+2, 3.020000E+2, 3.110000E+2,
        3.200000E+2, 3.290000E+2, 3.380000E+2, 3.470000E+2, 3.560000E+2,
        3.650000E+2, 3.740000E+2, 3.830000E+2, 3.920000E+2, 4.010000E+2,
        4.100000E+2, 4.190000E+2, 4.280000E+2, 4.370000E+2, 4.460000E+2,
        4.550000E+2, 4.640000E+2, 4.730000E+2, 4.820000E+2, 4.910000E+2,
        5.000000E+2, 5.090000E+2, 5.180000E+2, 5.270000E+2, 5.360000E+2,
        5.450000E+2, 5.540000E+2, 5.630000E+2, 5.720000E+2, 5.810000E+2,
        5.900000E+2, 5.970000E+2, 6.040000E+2, 6.110000E+2, 6.180000E+2,
        6.250000E+2, 6.329000E+2,)


betas = (
        0.000000E+0, 5.000000E-3, 1.000000E-2, 2.500000E-2, 5.000000E-2,
        7.500000E-2, 1.000000E-1, 1.500000E-1, 2.000000E-1, 2.500000E-1,
        3.000000E-1, 3.500000E-1, 4.000000E-1, 4.500000E-1, 5.000000E-1,
        5.500000E-1, 6.000000E-1, 6.500000E-1, 7.000000E-1, 7.500000E-1,
        8.000000E-1, 8.500000E-1, 9.000000E-1, 9.500000E-1, 1.000000E+0,
        1.050000E+0, 1.100000E+0, 1.150000E+0, 1.200000E+0, 1.250000E+0,
        1.300000E+0, 1.350000E+0, 1.400000E+0, 1.450000E+0, 1.500000E+0,
        1.550000E+0, 1.600000E+0, 1.650000E+0, 1.700000E+0, 1.750000E+0,
        1.800000E+0, 1.850000E+0, 1.900000E+0, 1.950000E+0, 2.000000E+0,
        2.050000E+0, 2.100000E+0, 2.150000E+0, 2.200000E+0, 2.250000E+0,
        2.300000E+0, 2.350000E+0, 2.400000E+0, 2.450000E+0, 2.500000E+0,
        2.550000E+0, 2.600000E+0, 2.650000E+0, 2.712700E+0, 2.770000E+0,
        2.830000E+0, 2.900000E+0, 2.960000E+0, 3.030000E+0, 3.110000E+0,
        3.180000E+0, 3.260000E+0, 3.340000E+0, 3.430000E+0, 3.520000E+0,
        3.610000E+0, 3.710000E+0, 3.810000E+0, 3.920000E+0, 4.030000E+0,
        4.140000E+0, 4.260000E+0, 4.390000E+0, 4.520000E+0, 4.650000E+0,
        4.800000E+0, 4.940000E+0, 5.100000E+0, 5.260000E+0, 5.425500E+0,
        5.600000E+0, 5.780000E+0, 5.970000E+0, 6.170000E+0, 6.370000E+0,
        6.590000E+0, 6.810000E+0, 7.040000E+0, 7.290000E+0, 7.540000E+0,
        7.810000E+0, 8.103000E+0, 8.370000E+0, 8.670000E+0, 8.980000E+0,
        9.300000E+0, 9.640000E+0, 1.000000E+1, 1.040000E+1, 1.081520E+1,
        1.116000E+1, 1.157000E+1, 1.200000E+1, 1.246000E+1, 1.298000E+1,
        1.352800E+1, 1.394000E+1, 1.448000E+1, 1.503000E+1, 1.562000E+1,
        1.620510E+1, 1.680000E+1, 1.723300E+1, 1.820000E+1, 1.892000E+1,
        1.940000E+1, 1.995000E+1, 2.070000E+1, 2.163000E+1, 2.210000E+1,
        2.266000E+1, 2.350000E+1, 2.430800E+1, 2.480000E+1, 2.534000E+1,
        2.620000E+1, 2.702000E+1, 2.750000E+1, 2.805000E+1, 2.890000E+1,
        2.973000E+1, 3.020000E+1, 3.076000E+1, 3.150000E+1, 3.241000E+1,
        3.290000E+1, 3.344000E+1, 3.400000E+1, 3.446600E+1, 3.530000E+1,
        3.615000E+1, 3.660000E+1, 3.718000E+1, 3.790000E+1, 3.880000E+1,
        3.989000E+1, 4.020000E+1, 4.051300E+1, 4.100000E+1, 4.154000E+1,
        4.200000E+1, 4.257000E+1, 4.320000E+1, 4.420000E+1, 4.528000E+1,
        4.600000E+1, 4.700000E+1, 4.799000E+1, 4.830000E+1, 4.861500E+1,
        4.960000E+1, 5.067000E+1, 5.120000E+1, 5.170000E+1, 5.250000E+1,
        5.338000E+1, 5.390000E+1, 5.441000E+1, 5.520000E+1, 5.600000E+1,
        5.672000E+1, 5.712000E+1, 5.840000E+1, 5.980000E+1, 6.120000E+1,
        6.251000E+1, 6.380000E+1, 6.523000E+1, 6.650000E+1, 6.790000E+1,
        6.840000E+1, 6.893000E+1, 6.980000E+1, 7.061000E+1, 7.110000E+1,
        7.164000E+1, 7.220000E+1, 7.292000E+1, 7.333400E+1, 7.400000E+1,
        7.480000E+1, 7.560000E+1, 7.640000E+1, 7.720000E+1, 7.800000E+1,
        7.890000E+1, 7.980000E+1, 8.070000E+1, 8.160000E+1, 8.250000E+1,
        8.340000E+1, 8.430000E+1, 8.520000E+1, 8.610000E+1, 8.700000E+1,
        8.800000E+1, 8.900000E+1, 9.000000E+1, 9.100000E+1, 9.200000E+1,
        9.300000E+1, 9.400000E+1, 9.500000E+1, 9.600000E+1, 9.700000E+1,
        9.800000E+1, 9.900000E+1, 1.000000E+2, 1.012000E+2, 1.024000E+2,
        1.036000E+2, 1.048000E+2, 1.060000E+2, 1.072000E+2, 1.084000E+2,
        1.096000E+2, 1.108000E+2, 1.120000E+2, 1.135000E+2, 1.150000E+2,
        1.165000E+2, 1.180000E+2, 1.195000E+2, 1.210000E+2, 1.225000E+2,
        1.240000E+2, 1.255000E+2, 1.270000E+2, 1.285000E+2, 1.300000E+2,
        1.320000E+2, 1.340000E+2, 1.360000E+2, 1.380000E+2, 1.400000E+2,
        1.420000E+2, 1.440000E+2, 1.460000E+2, 1.480000E+2, 1.500000E+2,
        1.520000E+2, 1.540000E+2, 1.560000E+2, 1.581000E+2,)


temperatures = [293.6, 323.6, 373.6, 423.6, 473.6, 523.6, 573.6, 647.2]

if __name__ == '__main__':
    args = parse_arguments()
    if args.subcommand == 'plot_pdos':
        plot_pdos(args)
    elif args.subcommand == 'leapr':
        run_leapr(args)
    elif args.subcommand == 'parse_mf7':
        parse_file7(args.input_file)
    else:
        pass
