#!/bin/bash
scPDB="../scPDB"
traintest_prots="scPDB_clustering/training_testing_subset.txt"
val_prots="scPDB_clustering/validation_subset.txt"
mkdir train_test_subset
while read line; do
   cp -r $(find $scPDB -name $line* -type d) train_test_subset 
done < $traintest_prots 

mkdir val_subset
while read line; do
   cp -r $(find $scPDB -name $line* -type d) val_subset 
done < $val_prots 


