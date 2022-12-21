conda activate wisp
# for folder in /scratch/iamerich/nics_dhf/dhf/testobj/*/*
for folder in /ubc/cs/research/kmyi/dhf/train_greedy_all/*/part0/mesh.obj
do  
    echo $folder
    for num_lod in 1 2 3 4 5 6
    do
        WISP_HEADLESS=1 python3 app/main.py --config configs/nglod_sdf.yaml --dataset-path $folder --epochs 100 --num-lods $num_lod
    done    
done