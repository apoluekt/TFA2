import uproot
import numpy as np
import amplitf.interface as atfi


def write_tuple(rootfile, array, branches, tree="tree"):
    """
    Store numpy 2D array in the ROOT file using uproot.
      rootfile : ROOT file name
      array : numpy array to store. The shape of the array should be (N, V),
              where N is the number of events in the NTuple, and V is the
              number of branches
      branches : list of V strings defining branch names
      tree : name of the tree
    All branches are of double precision
    """
    #with uproot.recreate(rootfile, compression=uproot.ZLIB(4)) as file:
    #    file[tree] = uproot.newtree({b: "float64" for b in branches})
    #    d = {b: array[:, i] for i, b in enumerate(branches)}
    #    # print(d)
    #    file[tree].extend(d)
    with uproot.recreate(rootfile, compression=uproot.ZLIB(4)) as file :
        file[tree] = {b: array[:, i] for i, b in enumerate(branches)}


def read_tuple(rootfile, branches, tree="tree"):
    """
    Load the contents of the tree from the ROOT file into numpy array.
    """
    with uproot.open(f"{rootfile}:{tree}") as t:
        a = [t[b].array(library="np") for b in branches]
    return atfi.const(np.stack(a, axis=1))


def read_tuple_filtered(
    rootfile, branches=None, tree="tree", selection=None, sel_branches=[]
):
    """
    Load the contents of the tree from the ROOT file into numpy array,
    applying the selection to each entry.
    """
    arrays = []
    with uproot.open(f"{rootfile}:{tree}") as t:
        if branches is None:
            read_branches = store_branches = t.keys()
        else:
            read_branches = branches + sel_branches
            store_branches = branches
        for data in t.iterate(branches=read_branches, library="pd"):
            if selection:
                df = data.query(selection)
            else:
                df = data
            arrays += [df[list(store_branches)].to_numpy()]
    if branches is None:
        return np.concatenate(arrays, axis=0), read_branches
    else:
        return np.concatenate(arrays, axis=0)
