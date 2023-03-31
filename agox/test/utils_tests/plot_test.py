from typing import List

import ase.data.colors
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase.cell import Cell
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.path import Path

from agox.candidates import StandardCandidate
from agox.test.test_utils import environment_and_dataset
from agox.utils.plot import plane_to_indices, plot_atoms, plot_cell


@pytest.mark.parametrize('plane,expected', [
    ('xy', (0, 1, 2, 1)),
    ('xy+', (0, 1, 2, 1)),
    ('xy-', (0, 1, 2, -1)),
    ('xz', (0, 2, 1, -1)),
    ('xz-', (0, 2, 1, 1)),
    ('yz+', (1, 2, 0, 1)),
    ('yx', (1, 0, 2, -1))
])
def test_plane_to_indices(plane, expected):
    indices = plane_to_indices(plane)
    assert indices == expected


@pytest.mark.parametrize('plane', [None, '', 'ab', 'abc', 'xx', 'xyz'])
def test_invalid_plane_to_indices(plane):
    with pytest.raises(ValueError, match='Invalid plane specification'):
        plane_to_indices(plane)


@pytest.mark.parametrize('plane', ['xy', 'xz', 'yz'])
@pytest.mark.parametrize('repeat', [0, 1])
def test_plot_atoms(environment_and_dataset, plane, repeat):
    matplotlib.use('Agg')

    _, dataset = environment_and_dataset
    structure: StandardCandidate = dataset[0]

    d0, d1, d2, kv = plane_to_indices(plane)

    sorted_numbers = structure.numbers[np.argsort(kv * structure.positions[:, d2])]

    fig, ax = plt.subplots()
    pcs = plot_atoms(ax,
                     structure,
                     plane=plane,
                     repeat=repeat,
                     repeat_patch_kwargs=dict(alpha=0.5))

    expected_repeat = repeat > 0 and structure.get_pbc()[[d0, d1]].sum() > 0

    assert len(ax.collections) == len(pcs)

    if not expected_repeat:
        assert len(ax.collections) == 1
    else:
        assert len(ax.collections) == 2

    assert isinstance(ax.collections[0], PatchCollection)
    p: PatchCollection = ax.collections[0]

    paths: List[Path] = p.get_paths()
    assert len(paths) == len(structure)

    colors = p.get_facecolor()[:, :3]
    alphas = p.get_facecolor()[:, 3]
    np.testing.assert_allclose(colors, ase.data.colors.jmol_colors[sorted_numbers])
    np.testing.assert_allclose(alphas, 1)

    if expected_repeat:
        pbc_dimensions = structure.get_pbc()[[d0, d1]].sum()
        expected_copies = (2 * repeat + 1) ** pbc_dimensions - 1

        assert isinstance(ax.collections[1], PatchCollection)
        p: PatchCollection = ax.collections[1]

        paths: List[Path] = p.get_paths()
        assert len(paths) == expected_copies * len(structure)

        alphas = p.get_facecolor()[:, 3]
        np.testing.assert_allclose(alphas, np.full(alphas.shape, 0.5))

    ax.autoscale_view()

    ax_xmin, ax_xmax = ax.get_xlim()
    ax_ymin, ax_ymax = ax.get_ylim()

    struc_xmin, struc_ymin = structure.positions[:, [d0, d1]].min(axis=0)
    struc_xmax, struc_ymax = structure.positions[:, [d0, d1]].max(axis=0)
    struc_xspan = struc_xmax - struc_xmin
    struc_yspan = struc_ymax - struc_ymin

    assert struc_xmin - struc_xspan < ax_xmin <= struc_xmin
    assert struc_xmax + struc_xspan > ax_xmax >= struc_xmax
    assert struc_ymin - struc_yspan < ax_ymin <= struc_ymin
    assert struc_ymax + struc_yspan > ax_ymax >= struc_ymax

    plt.close(fig)


@pytest.mark.parametrize('plane', ['xy', 'xz', 'yz'])
def test_plot_cell(environment_and_dataset, plane):
    matplotlib.use('Agg')

    _, dataset = environment_and_dataset
    cell: Cell = dataset[0].cell

    fig, ax = plt.subplots()
    plot_cell(ax, cell, plane=plane)

    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], PolyCollection)
    p: PolyCollection = ax.collections[0]

    paths: List[Path] = p.get_paths()
    assert len(paths) == 6

    plt.close(fig)
