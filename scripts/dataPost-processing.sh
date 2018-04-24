#!/bin/bash
# author: Hao WANG
TASK=zh-ja
SCRIPT=/itigo/files/Tools/accessByOldOrganization/Scripts/wat2017/script.segmentation.distribution
MOSES_SCRIPT=/itigo/files/Tools/accessByOldOrganization/TranslationEngines/mosesdecoder-RELEASE-2.1.1/scripts
#Evaluation
EVAL_SCRIPTS=/itigo/files/Tools/accessByOldOrganization/MTEvaluation
RIBES=/itigo/files/Tools/accessByOldOrganization/TransaltionEvaluation/RIBES-1.03.1
OUTPUT=$1

mkdir -p ${OUTPUT}/eval
cp ${OUTPUT}/pred.txt ${OUTPUT}/eval/test.ja
cp /itigo/Uploads/ASPEC-JC.clean/Juman+Stanford/test.ja ${OUTPUT}/eval/ref.ja

cd ${OUTPUT}/eval

for file in test ref; do
    cat ${file}.ja | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' | \
    sh ${SCRIPT}/remove-space.sh | \
    perl ${SCRIPT}/h2z-utf8-without-space.pl | \
    juman -b | \
    perl -ne 'chomp; if($_ eq "EOS"){print join(" ",@b),"\n"; @b=();} else {@a=split/ /; push @b, $a[0];}' | \
    perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
    perl -Mencoding=utf8 -pe 'while(s/([Ａ-Ｚ]) ([Ａ-Ｚａ-ｚ])/$1$2/g){} while(s/([ａ-ｚ]) ([ａ-ｚ])/$1$2/g){}' \
    > ${file}.${TASK}.juman.ja
done

# For BLEU
perl ${MOSES_SCRIPT}/generic/multi-bleu.perl ref.${TASK}.juman.ja < test.${TASK}.juman.ja
# For RIBES
python3 ${RIBES}/RIBES.py -c -r ref.${TASK}.juman.ja test.${TASK}.juman.ja

python2 ${EVAL_SCRIPTS}/mteval.py ref.${TASK}.juman.ja < test.${TASK}.juman.ja
