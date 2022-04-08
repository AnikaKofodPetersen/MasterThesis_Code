#!/usr/bin/env python3

#
# Converts ESM resiude tensor .pt files to their PCA-reduced equivalent
# E.g. 500 seqs x positions x 1280 -> 500 x positions x 30
# by Magnus Høie, maghoi@dtu.dk
# Adaptations for specific usecase by Anika K. Petersen
#

#Imports
import os, glob
from collections import OrderedDict
import numpy as np
import torch
import pickle as pk
from Bio import SeqIO
import argparse
import random
import sys

#Parser
parser = argparse.ArgumentParser(description='Convert folder of ESM residue tensor .pt files to output folder with their PCA-reduced equivalent')
parser.add_argument("-i", '--embedding_dir', dest="EMBEDDING_DIR", default="seqs_embeddings/",
                    help='ESM embedding directory with label.pt files')
parser.add_argument("-o", '--output_dir', dest="OUTDIR", default="output/",
                    help='ESM embedding directory with label.pt files')
parser.add_argument("-d", '--dimension', dest="DIM", default=30,
                    help='Target dimensionality')
parser.add_argument("-s", '--size', dest="SIZE", default=1280,
                    help='input dimensionality')
parser.add_argument("-e", '--esm', dest="ESM", default=True,
                    help='Esm or PS embeddings')
parser.add_argument("-v", '--verbose', dest="VERBOSE", default=2,
                    help='Verbose printing')
args = parser.parse_args()


def data_split(embdir=args.EMBEDDING_DIR,n=3700):
    """ split data intow chuncks of ~4000"""
    files = glob.glob(args.EMBEDDING_DIR + "*.pt")
    random.shuffle(files)
    out = [files[i:i+n] for i in range(0, len(files), n)]
    return out

def stack_res_tensors(tensors, embedding_dim=int(args.SIZE)):
    # Prepare to stack tensors
    n_seqs = len(tensors)
    seq_max_length = max([t.shape[0] for t in tensors])

    # Initialize empty padded vector, with 0-padding for sequences with less than max length residues
    fill_tensor = torch.zeros(size=(n_seqs, seq_max_length, embedding_dim))
    
    # Load torch tensors from ESM embeddings matching sequence, fill padded tensor
    for i, tensor in enumerate(tensors):
        fill_tensor[i, 0:tensor.shape[0]] = tensor

    return(fill_tensor)

def PCA_reduce_stacked_tensor(stacked_tensor, pca_dim=int(args.DIM), first = True):
     """
     Fits PCA and transforms on flattened dataset representing all positions x ESM_embeddings, before reshaping back to n_seqs x positions x pca_dim

     Returns:
     (pca_stacked_tensor) [torch.Tensor]: Dataset reduced to n_seqs x sequence_length x pca_dim (30)
     """

     from sklearn.decomposition import PCA
     pca = PCA(pca_dim)

     # Flatten training dataset to shape n_positions/residues x embedding_dim
     positions = stacked_tensor.view( stacked_tensor.shape[0]*stacked_tensor.shape[1], int(args.SIZE) )

     # PCA reduce to n_dim (30) dimensions along positions
     if first == True:
         positions_reduced = pca.fit_transform(positions)
         explained = pca.explained_variance_ratio_
         pk.dump(pca, open("pca.pkl","wb"))
     else:
         pca_reload = pk.load(open("pca.pkl",'rb'))
         positions_reduced = pca_reload.transform(positions)
         explained = None
         
     # Reshape dataset to n_seqs x n_positions x 30
     pca_stacked_tensor = positions_reduced.reshape( stacked_tensor.shape[0], stacked_tensor.shape[1], pca_dim )
     pca_stacked_tensor = torch.Tensor(pca_stacked_tensor)

     return[pca_stacked_tensor, explained]

def save_processed_tensor(tensor_proc, id_tensor_odict, outdir, verbose=0, esm = args.ESM):

        # Create new dict with original shape of PCA reduced tensors (remove zero-padding in processed stacked tensor)
        pca_dict = {}
        for i, t_id in enumerate(id_tensor_odict.keys()):
                orig_tensor = id_tensor_odict[t_id]
                pca_tensor = tensor_proc[i]
                
                # Fill with PCA reduced tensor, preserving original shape
                pca_dict[t_id] = pca_tensor[0:orig_tensor.shape[0], 0:orig_tensor.shape[1]]

        # Save
        print("Writing PCA-reduced tensors to %s" % outdir)
        os.makedirs(outdir, exist_ok=True)
        if esm == "True":
            for t_id, t in pca_dict.items():
                outpath = outdir + "ESM_" + str(t_id) + ".pt"

                #if verbose: print("Writing ID %s %s to %s" % (t_id, t.shape, outpath))

                # NB: tensor.clone() ensures tensor is stored as its actual slice and not as 37 mb tensor represented in memory
                # Difference of e.g. 20 kb vs 37 mb
                torch.save(t.clone(), outpath)
        else:
            for t_id, t in pca_dict.items():
                outpath = outdir + "PS_" + str(t_id) + ".pt"

                #if verbose: print("Writing ID %s %s to %s" % (t_id, t.shape, outpath))

                # NB: tensor.clone() ensures tensor is stored as its actual slice and not as 37 mb tensor represented in memory
                # Difference of e.g. 20 kb vs 37 mb
                torch.save(t.clone(), outpath)        

        return pca_dict
                        
if __name__ == "__main__":   
        #Partition data
        parts = data_split(embdir=args.EMBEDDING_DIR,n=5000)
        first = True
        tensor_files = glob.glob(args.EMBEDDING_DIR + "*.pt")
        if args.VERBOSE: print("Found %s tensor files in %s" % (len(tensor_files), args.EMBEDDING_DIR))
        print(f"working with 5000 files at a time: total of {len(parts)} partitions")
        
        part_count = 0
        for part in parts:
            # Get ESM in ordered dict
            id_tensor_odict = OrderedDict()
            part_count += 1
            print(f"part {part_count}")
            part_files = list(set(tensor_files) & set(part))
            
            for filex in part_files:
                esm_d = torch.load(filex)
                if args.ESM == "True":
                    id_tensor_odict[esm_d["label"]] = esm_d["representations"][33]
                else:
                    name = filex.split("_")[-1]
                    name = name.split(".")[0]
                    id_tensor_odict[name] = esm_d.detach()
                
            # Stack with zero-padding by longest sequence length
            tensors = [tensor for tensor in id_tensor_odict.values()]
            stacked_tensor = stack_res_tensors(tensors)
            if args.VERBOSE: print("Stacked tensor:", stacked_tensor.shape)
                
            # PCA reduce to 30 dimensions
            if args.VERBOSE: print("Fitting PCA model to stacked_tensor ...")
            output =  PCA_reduce_stacked_tensor(stacked_tensor, pca_dim=int(args.DIM), first = first)
            if args.VERBOSE: print("done")
            pca_stacked_tensor = output[0]
            explained = output[1]
            
            if first == True:
                with open(f"explaned_variance_ESM={args.ESM}.txt", "w+") as outfile:
                    outfile.write(str(explained))
        
            if args.VERBOSE: print("PCA stacked tensor:", pca_stacked_tensor.shape)


            # Save to new files
            pca_dict = save_processed_tensor(pca_stacked_tensor, id_tensor_odict, outdir=args.OUTDIR, verbose=args.VERBOSE)
            first = False

