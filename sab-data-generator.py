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
        pattern = re.compile(r'\d([+-])')
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
                line = sab_file.readline()
                doubles = [
                        to_float(line[start:start+11])
                        for start in range(0, 66, 11)
                        if not line[start:start+11].isspace()]
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
                    for S in [
                            to_float(line[start:start+11])
                            for start in range(0, 66, 11)
                            if not line[start:start+11].isspace()]:
                        alphas.append(next(unique_alphas))
                        betas.append(beta)
                        Ts.append(temp)
                        Ss.append(S)
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
        5.0000e-04, 7.5000e-02, 3.0000e-01, 4.5000e-01, 6.1000e-01,
        8.8000e-01, 1.3300e+00, 2.0900e+00, 3.3800e+00, 5.4255e+00,
        8.8000e+00, 1.3528e+01, 1.8920e+01, 2.7020e+01, 3.4466e+01,
        4.2570e+01, 5.1200e+01, 5.9800e+01, 6.7900e+01, 8.0000e+01,
        1.1300e+02, 1.5400e+02, 1.9900e+02, 2.5500e+02, 3.1100e+02,
        3.6500e+02, 4.1900e+02, 4.7300e+02, 5.2700e+02, 5.8100e+02,
        6.2500e+02)

betas = (
        0.0000e+00, 1.0000e-01, 4.0000e-01, 7.0000e-01, 1.0000e+00,
        1.3000e+00, 1.6000e+00, 1.9000e+00, 2.2000e+00, 2.5000e+00,
        2.8300e+00, 3.2600e+00, 3.8100e+00, 4.5200e+00, 5.4255e+00,
        6.5900e+00, 8.1030e+00, 1.0000e+01, 1.2460e+01, 1.5620e+01,
        1.9400e+01, 2.3500e+01, 2.7500e+01, 3.1500e+01, 3.5300e+01,
        3.9890e+01, 4.2570e+01, 4.7990e+01, 5.1700e+01, 5.6000e+01,
        6.2510e+01, 6.8930e+01, 7.2920e+01, 7.7200e+01, 8.2500e+01,
        8.8000e+01, 9.4000e+01, 1.0000e+02, 1.0720e+02, 1.1500e+02,
        1.2400e+02, 1.3400e+02, 1.4600e+02, 1.5810e+02)

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
