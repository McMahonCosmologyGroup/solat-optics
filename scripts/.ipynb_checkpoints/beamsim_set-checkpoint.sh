declare -a FREQ
FREQ=('150')
len=${#FREQ[@]}

for (( i=0; i<$len; i++ ))
do
    mpiexec -n 70 python3 holog_run.py ${FREQ[i]}

done