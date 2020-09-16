import uproot
import numpy as np

def write_tuple(rootfile, array, branches, tree="tree") : 
  with uproot.recreate(rootfile, compression=uproot.ZLIB(4)) as file :  
    file[tree] = uproot.newtree( { b : "float64" for b in branches } )
    d = { b : array[:,i] for i,b in enumerate(branches) }
    #print(d)
    file[tree].extend(d)

def read_tuple(rootfile, branches, tree = "tree") : 
  with uproot.open(rootfile) as file : 
    t = file[tree]
    a = [ t.array(b) for b in branches ]
  return np.stack(a, axis = 1)

def read_tuple_filtered(rootfile, branches, tree = "tree", selection = None, sel_branches = []) : 
  arrays = []
  with uproot.open(rootfile) as file : 
    t = file[tree]
    for data in t.pandas.iterate(branches = branches + sel_branches) : 
      if selection : df = data.query(selection)
      else : df = data
      arrays += [ df[list(branches)].to_numpy() ]
  return np.concatenate(arrays, axis = 0)
