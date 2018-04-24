#!/bin/bash
# author: Hao WANG

LANG1=$1
LANG2=$2
EMBEDED_DIM=500
TASK=$1-$2
word2vec=/itigo/files/Tools/accessByOldOrganization/Word2Vec/word2vec
Data=/itigo/Uploads/ASPEC-JC.clean/BPE



$word2vec -train $Data/train.$LANG1 -size ${EMBEDED_DIM} -output w2v.$LANG1

$word2vec -train $Data/train.$LANG2 -size ${EMBEDED_DIM} -output w2v.$LANG2
