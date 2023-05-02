import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import ase.data
import ase.data.colors
import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.patches import Circle
from numpy.typing import NDArray


def plot_atoms(ax: Axes,
               atoms: Atoms,
               *,
               plane: str = 'xy+',
               radius_factor: float = 1.0,
               repeat: int = 0,
               patch_kwargs: Optional[Dict[str, Any]] = None,
               repeat_patch_kwargs: Optional[Dict[str, Any]] = None) -> List[PatchCollection]:
    """Plot a 2D projection of an `Atoms` object into an `Axes` object.

    Parameters
    ----------
    ax : Axes
        Axes to plot the atoms in.
    atoms : Atoms
        Atoms to plot.
    plane : str, optional
        Plane to project the atoms on. See `plane_to_indices` for information
        on valid plane strings, by default 'xy+'.
    radius_factor : float, optional
        Factor to multiply the atomic covalent radii with, by default 1.0.
    repeat : int, optional
        Number of times to repeat the Atoms object in dimensions with periodic
        boundary conditions, by default 0. Repeated copies are not included in
        autoscaling. Note that if none of the two projection axes have periodic
        boundary conditions, this parameter is effectively ignored.
    patch_kwargs : Optional[Dict[str, Any]], optional
        Keyword arguments forwarded to the `Circle` constructor for each atom,
        by default None.
    repeat_patch_kwargs : Optional[Dict[str, Any]], optional
        Keyword arguments additionally forwarded to the `Circle` constructor
        for each repeated atom, by default None.

    Returns
    -------
    List[PatchCollection]
        List of PatchCollections that make up the Atoms object. If the Atoms
        object is not repeated, this list contains one element. If it is
        repeated, this list contains two elements: a patch collection
        containing the non-repeated patches and a patch collection containing
        the repeated patches.
    """

    if patch_kwargs is None:
        patch_kwargs = {}
    if repeat_patch_kwargs is None:
        repeat_patch_kwargs = {}

    d0, d1, d2, kv = plane_to_indices(plane)

    radii = radius_factor * ase.data.covalent_radii[atoms.numbers]
    colors = ase.data.colors.jmol_colors[atoms.numbers]

    if repeat > 0:
        atoms_ = atoms.copy()
        pbc = atoms.get_pbc()

        repeat_range_d0 = range(-repeat, repeat + 1) if pbc[d0] else [0]
        repeat_range_d1 = range(-repeat, repeat + 1) if pbc[d1] else [0]

        for x, y in itertools.product(repeat_range_d0, repeat_range_d1):
            if x == y == 0:
                continue
            offset_uc = np.zeros(3)
            offset_uc[[d0, d1]] = [x, y]
            offset = atoms.cell.cartesian_positions(offset_uc)
            repeat_atoms = atoms.copy()
            repeat_atoms.translate(offset)
            atoms_ += repeat_atoms
    else:
        atoms_ = atoms

    n_atoms = len(atoms)
    patches: List[Circle] = []
    repeat_patches: List[Circle] = []

    for i in np.argsort(kv * atoms_.positions[:, d2]):
        patch_kwargs_ = {**patch_kwargs} if i < n_atoms else {**patch_kwargs, **repeat_patch_kwargs}

        position = atoms_[i].position
        radius = radii[i % n_atoms]
        color = colors[i % n_atoms]

        circle = Circle(position[[d0, d1]], radius=radius, ec='k', fc=color, **patch_kwargs_)

        if i < n_atoms:
            patches.append(circle)
        else:
            repeat_patches.append(circle)

    pcs: List[PatchCollection] = []

    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    pcs.append(p)

    if len(repeat_patches) > 0:
        repeat_p = PatchCollection(repeat_patches, match_original=True)
        ax.add_collection(repeat_p, autolim=False)
        pcs.append(repeat_p)

    ax.set_aspect('equal')
    ax.tick_params(bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False,
                   labelleft=False, labelright=False)

    return pcs


def plot_cell(ax: Axes,
              cell: Union[Cell, NDArray],
              *,
              plane: str = 'xy+',
              offset: Optional[NDArray] = None,
              collection_kwargs: Dict[str, Any] = None):
    """Plot a 2D projection of a `Cell` object into an `Axes` object.

    Parameters
    ----------
    ax : Axes
        Axes to plot the cell in.
    cell : Union[Cell, NDArray]
        Cell to plot.
    plane : str, optional
        Plane to project the cell on. See `plane_to_indices` for information on
        valid plane strings, by default 'xy+'.
    offset : Optional[NDArray], optional
        Cartesian coordinates to apply as offset to the base of the cell, by
        default None (no offset).
    collection_kwargs : Dict[str, Any], optional
        Keyword arguments additionally forwarded to the `PolyCollection`
        constructor for the cell, by default None.

    Raises
    ------
    ValueError
        Raised if `offset` or `cell` has an incorrect shape.
    """

    default_collection_kwargs = dict(
        edgecolors='k',
        facecolors='none',
        linestyles='dotted'
    )

    if offset is None:
        offset = np.zeros(3)
    else:
        offset = np.asarray(offset)

    if offset.shape != (3,):
        raise ValueError(f'`offset` must have shape (3,), input has {offset.shape}')

    if isinstance(cell, Cell):
        cell = cell.complete()

    if cell.shape != (3, 3):
        raise ValueError(f'`cell` must have shape (3, 3), input has {cell.shape}')

    if collection_kwargs is None:
        collection_kwargs = {}

    collection_kwargs = {**default_collection_kwargs, **collection_kwargs}

    d0, d1, _, _ = plane_to_indices(plane)

    verts_uc = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # xy, z = 0
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],  # xy, z = 1
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # xz, y = 0
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # xz, y = 1
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # yz, x = 0
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]   # yz, x = 1
    ])
    verts_3d = verts_uc @ cell + offset
    verts_2d = verts_3d[:, :, [d0, d1]]

    p = PolyCollection(verts_2d, **collection_kwargs)
    ax.add_collection(p)


def plane_to_indices(plane: str) -> Tuple[int, int, int, int]:
    """Convert a string representation of a plane to a 4-tuple of indices
    usable to perform operations using this plane as a projection.

    Parameters
    ----------
    plane : str
        String representation of the plane, in the format /[xyz]{2}[+-]?/. The
        first two characters define the plane's Cartesian axes (and should not
        be equal), and the third optional character defines whether the
        perpendicular axis is positive (+) or negative (-); the default is
        positive.

        Examples of string representations: 'xy', 'xy+', 'xy-', 'xz', 'zx+'.

    Returns
    -------
    Tuple[int, int, int, int]
        A tuple (d0, d1, d2, kv) describing this plane:
        - d0: index of the first plane axis (0, 1, or 2);
        - d1: index of the second plane axis (0, 1, or 2);
        - d2: index of the axis perpendicular to the plane (0, 1, or 2);
        - kv: sign of the axis perpendicular to the plane (-1 or 1).

    Raises
    ------
    ValueError
        Raised if the plane specification is invalid.
    """
    if not isinstance(plane, str):
        plane = str(plane)

    if len(plane) < 2:
        raise ValueError(f'Invalid plane specification {plane}')

    I = np.eye(3)

    d0 = ord(plane[0]) - ord('x')  # 'x' -> 0, 'y' -> 1, 'z' -> 2
    d1 = ord(plane[1]) - ord('x')  # 'x' -> 0, 'y' -> 1, 'z' -> 2

    if len(plane) > 2:
        s = -(ord(plane[2]) - ord(','))  # '+' -> 1, '-' -> -1
    else:
        s = 1

    if d0 not in [0, 1, 2] or d1 not in [0, 1, 2] or d0 == d1 or s not in [-1, 1]:
        raise ValueError(f'Invalid plane specification {plane}')

    d2 = ({0, 1, 2} - {d0} - {d1}).pop()

    i, j = I[d0, :], I[d1, :]
    k = np.cross(i, j)
    kv = s * int(np.sum(k))  # -1 or 1

    return (d0, d1, d2, kv)
