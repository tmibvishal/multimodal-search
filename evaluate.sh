echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module () {
        eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
}

module load apps/anaconda/3
source activate InfoProject
module unload apps/anaconda/3

time python evaluation.py -i data/flickr8k/image_caption.txt -o temporary_directory/evaluation_bm25_8k.txt -c temporary_directory/collection_8k.pkl -q temporary_directory/queries_8k.pkl -f bm25
time python evaluation.py -i data/flickr8k/image_caption.txt -o temporary_directory/evaluation_vsm_8k.txt -c temporary_directory/collection_8k.pkl -q temporary_directory/queries_8k.pkl -f vsm
time python evaluation.py -i data/flickr8k/image_caption.txt -o temporary_directory/evaluation_lm_8k.txt -c temporary_directory/collection_8k.pkl -q temporary_directory/queries_8k.pkl -f lm