# Master thesis code 
### To make my thesis project, I followed this approach: 

**##########################################################** \
                   **0: DataPreprocessing**        \
**##########################################################** \
*0)*  
Inspired by the NetSolP approch, I went to their website and went to the [download-page.](https://services.healthtech.dtu.dk/service.php?NetSolP-1.0)  \
From here I downloaded the collected data (netsolp-1.0.ALL.tar.gz)and extracted the content.  \
Since the NESG dataset have been split, I choose to go to the [original source](https://loschmidt.chemi.muni.cz/soluprot/?page=download)
and download the original dataset.  \
The Psi-Biology dataset I've used the version delivered by [NetSolP](https://services.healthtech.dtu.dk/service.php?NetSolP-1.0)  \
*1)*  \
I then ran the notebook "0_DataCollectionAndPreprocessing".  \
This notebook loads the NESG and Psi-Bio datasets and cleans up the data. \ 
This means that it concatenates the datasets, removes duplicates and sequences with contradictorary labels.  \
I also removed sequences longer than 1000 AA, since I will be using ESM with a limit of 1024 AA.   \
\
\
**##########################################################** \
                 **1: Structures** \
**##########################################################** \
To avoid predicting too many structures, the AlphaFold database came in handy. \
It currently contains structure predictions of all UniProt proteins. \
*0)*  \
All accession numbers from the AlphaFold database were downloaded from: \
http://ftp.ebi.ac.uk/pub/databases/alphafold/accession_ids.txt \
*1)*  \
The ID's were extracted by running the notebook "1a_format_accession_ID_file". \
This produced the Accessions.txt file. \
*2)*  \
All uniprot fastafiles were downloaded by running: \
`$ for ID in $(cat Accessions.txt); do wget https://www.uniprot.org/uniprot/$ID.fasta; done` \
*3)* \
The downloaded fastafiles were concatenated into suitable sizes of databases, and databases where made by running: \
`$ makeblastdb -in X -parse_seqids -blastdb_version 5  -title "alphafold_DB" -dbtype prot  -max_file_sz '3GB' -out X_DB` \
X being the name of the partition. Due to the large database size, it was not possible to run this in one single go, creating one single database \
*4)* \
To find if the the protein data in use already had AlphaFold structures, a protein blast was performed against each of the newly made databases: \
`$ blastp -query ../bulk.fasta -db X_DB/X_DB -outfmt 6 -qcov_hsp_perc 80 -num_threads 4 -out ../blast_output/blastp_X.txt` \
*5)*   \
The blast output was analyzed using the notebook "1b_blast_analysis". \
 \
 \
**##########################################################** \
                 **2: Embeddings** \
**##########################################################** \
*0)* \
I used the "2a_ESMEmbeddings" notebook to make a bulk fasta file and esm embeddings \
*1)*  \
I used the "2b_ProteinSolverEmbeddings" notebook to make ProteinSolver embeddings \
*2)* \
I used the approach of "2c_PCA" notebook to lower the dimensionality of the embeddings \
*Note: due to lacking processing power of my local machine, the PCA of the embeddings were performed on the computerome cluster at DTU* \
*3)* \
Since I did not have a 100% match filter on the structures (80% query coverage, 90% ID and 80% subject coverage), some of the embeddings don't have the same dimension due to gaps in the alignment. \
To overcome this, I have to add zeroes where the alignment indicates a gap.  \
I did so by using the notebook "2d_AddGapZeros" \
 \
 \
**##########################################################** \
                 **3: PreTraining** \
**##########################################################** \
*0)*  \
Clustering was perfromed using MMseqs2 with an identity threshold of 50% and alignment coverage of 80%. \
this was done by the following commands: \
`$ mmseqs createdb bulk.fasta DB` \
`$ mmseqs cluster DB DB_clu tmp --min-seq-id 50` \
`$ mmseqs createtsv DB DB DB_clu DB_clu_50_id.tsv` \
*1)* \
In order to explore the model performance, different learning rates and the use of sub-datasets, the scripts in "3a_hyperparameters" were run on the computerome cluster at DTU.  \
*2)* \
The final model was pre-trained on all the available data as shown in the notebook "3b_MultitaskTraining". \
 \
 \
**##########################################################** \
                **4: FineTuning** \
**##########################################################** \
*0)* \
To preprocess the fine-tuning data in a similar manner to the pre-training data, the notebooks 4a-4d were run. \
*1)* \
I used the notebook "4e_FuneTuning" to fine-tune the model on the fine-tuning dataset. \
 \
 \
**##########################################################** \
                 **5: Testing** \
**##########################################################**  \
*0)* \
To see if the model was able to learn solely from the fine-tuning dataset, the notebook "5a_FineTuning-JainOnly" was used. \
*1)* \
I used the notebook "5b_Comparison" to test other aggregation predictors on the fine-tuning dataset, in order to compare performances. \
*2)* \
I used the notebook "5c_RFC" to test the performances of Random Forest Classifiers on the fine-tuning data \
