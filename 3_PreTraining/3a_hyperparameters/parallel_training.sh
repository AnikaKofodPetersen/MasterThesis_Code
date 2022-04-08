#!/bin/bash
#Anika Kofod Petersen
#Run 20/3-22
for i in 1 2 3 4
do
    cd /home/projects/ht3_aim/people/anipet/new_model_training/
    echo "module load tools ngs anaconda3/4.4.0; python3 /home/projects/ht3_aim/people/anipet/new_model_training/Training${i}.py" | qsub -W group_list=ht3_aim -A ht3_aim -l nodes=1:ppn=35:thinnode,mem=30gb,walltime="03:00:00:00"
done
