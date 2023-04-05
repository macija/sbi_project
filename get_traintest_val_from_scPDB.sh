#!/bin/bash
scPDB="../TrainData/scPDB"
traintest_prots="scPDB_clustering/families_1rep.txt"
val_prots="scPDB_clustering/validation_subset.txt"
mkdir ../TrainData/train_test_families_1rep
while read line; do
   cp -r $(find $scPDB -name $line* -type d) ../TrainData/train_test_families_1rep 
done < $traintest_prots 

#mkdir val_subset
#while read line; do
#   cp -r $(find $scPDB -name $line* -type d) val_subset 
#done < $val_prots 


