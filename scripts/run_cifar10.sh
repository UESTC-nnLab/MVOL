IN_DATASET=CIFAR-10

# Single Model Setting
for seed in 0 1000 2000 3000 4000
do
    echo "seed = $seed"
    python train_vanilla.py --exp-cat-name vanilla --model-arch wideresnet\
           --name vanilla_seed"$seed" --in-dataset $IN_DATASET --gpu 0 --droprate 0.0 --seed "$seed"\
           --depth 40 --width 2 --batch-size 128
    python eval_ood_detection.py --exp-cat-name vanilla --if-aux 0 --model-arch wideresnet\
           --epochs 100 --in-dataset $IN_DATASET --name vanilla_seed"$seed" --method maxlogit --gpu 0  
    python compute_metrics.py --exp-cat-name vanilla --if-aux 0 --model-arch wideresnet\
           --in-dataset $IN_DATASET --name vanilla_seed"$seed" --method maxlogit
done


for seed in 0 1000 2000 3000 4000
do
    echo "seed = $seed"
    python train_oe.py --exp-cat-name oe --auxiliary-dataset RandomImages300k --model-arch wideresnet\
            --name oe_seed"$seed" --in-dataset $IN_DATASET --gpu 0 --seed "$seed" --beta 0.5 --droprate 0.0\
            --batch-size 128 --ood-batch-size 128
    python eval_ood_detection.py --exp-cat-name oe --if-aux 1 --auxiliary-dataset RandomImages300k --model-arch wideresnet\
            --epochs 100 --in-dataset $IN_DATASET --name oe_seed"$seed" --method maxlogit --gpu 0  
    python compute_metrics.py --exp-cat-name oe --if-aux 1 --auxiliary-dataset RandomImages300k --model-arch wideresnet\
            --in-dataset $IN_DATASET --name oe_seed"$seed" --method maxlogit
done


for seed in 0 1000 2000 3000 4000
do    
    echo "seed = $seed"
    python train_mvol.py --exp-cat-name mvol --auxiliary-dataset RandomImages300k --model-arch wideresnet\
            --name mvol_seed"$seed" --in-dataset $IN_DATASET --gpu 0 --seed "$seed" --beta 0.5 --droprate 0.0\
            --batch-size 128 --ood-batch-size 128 --factor 1
    python eval_ood_detection.py --exp-cat-name mvol --if-aux 1 --auxiliary-dataset RandomImages300k --model-arch wideresnet\
            --epochs 100 --in-dataset $IN_DATASET --name mvol_seed"$seed" --method maxlogit --gpu 0  
    python compute_metrics.py --exp-cat-name mvol --if-aux 1 --auxiliary-dataset RandomImages300k --model-arch wideresnet\
            --in-dataset $IN_DATASET --name mvol_seed"$seed" --method maxlogit
done

# Ensemble Distilltion Model Setting
for seed in 0 1000 2000 3000 4000
do
    echo "seed = $seed"
    python train_vanilla_ensemble.py \
    --exp-cat-name vanilla_ensemble --ekd-name vanilla_ensemble_seed"$seed" --model-arch wideresnet\
    --teacher-names 'vanilla_seed0,vanilla_seed1000,vanilla_seed2000,vanilla_seed3000,vanilla_seed4000,vanilla_seed5000,vanilla_seed6000,vanilla_seed7000,vanilla_seed8000,vanilla_seed9000'\
    --in-dataset $IN_DATASET --gpu $GPU --seed 0 --alpha 0.1 --temperature 2 --depth 40 --width 2 --droprate 0.0 --batch-size 128
    python eval_ood_detection.py --exp-cat-name vanilla_ensemble --if-aux 0 --epochs 100 --in-dataset $IN_DATASET --name vanilla_ensemble_seed"$seed" --method maxlogit --gpu $GPU --model-arch wideresnet
    python compute_metrics.py --exp-cat-name vanilla_ensemble --if-aux 0 --in-dataset $IN_DATASET --name vanilla_ensemble_seed"$seed" --method msp --model-arch wideresnet
done


for seed in 0 1000 2000 3000 4000
do
    python train_oe_ensemble.py --exp-cat-name oe_ensemble --model-arch wideresnet --ekd-name oe_ensemble_seed"$seed"_record \
        --in-dataset $IN_DATASET --auxiliary-dataset RandomImages300k \
        --batch-size 128 --ood-batch-size 128 --gpu 0 --seed $seed --alpha 0.1 --temperature 2 --beta 0.5 --depth 40 --width 2 --save-epoch 1\
        --teacher-names 'vanilla_seed0,vanilla_seed1000,vanilla_seed2000,vanilla_seed3000,vanilla_seed4000,vanilla_seed5000,vanilla_seed6000,vanilla_seed7000,vanilla_seed8000,vanilla_seed9000'
    python eval_ood_detection.py --if-aux 1 --exp-cat-name oe_ensemble --epochs 100 --in-dataset CIFAR-10 --model-arch wideresnet  --name oe_ensemble_seed"$seed" --method maxlogit --gpu 0 --depth 40 --width 2
    python compute_metrics.py --if-aux 1 --exp-cat-name oe_ensemble --in-dataset CIFAR-100 --name oe_ensemble_seed"$seed" --method maxlogit --model-arch wideresnet 
done


for seed in 0 1000 2000 3000 4000
do
    python train_mvol_ensemble.py --exp-cat-name multiview_mvol --model-arch wideresnet\
        --ekd-name mvol_ensemble_seed"$seed" \
        --in-dataset $IN_DATASET --auxiliary-dataset RandomImages300k\
        --factor 1.0 --batch-size 128 --ood-batch-size 128 --gpu $GPU --seed $seed --alpha 0.1 --temperature 2 --beta 0.5 --depth 40 --width 2\
        --teacher-names 'vanilla_seed0,vanilla_seed1000,vanilla_seed2000,vanilla_seed3000,vanilla_seed4000,vanilla_seed5000,vanilla_seed6000,vanilla_seed7000,vanilla_seed8000,vanilla_seed9000'
    python eval_ood_detection.py --if-aux 1 --auxiliary-dataset RandomImages300k --exp-cat-name mvol_ensemble --epochs 100 --in-dataset $IN_DATASET --model-arch wideresnet\
        --name mvol_ensemble_seed"$seed" --method maxlogit --gpu $GPU --depth 40 --width 2
    python compute_metrics.py --if-aux 1 --exp-cat-name mvol_ensemble --in-dataset $IN_DATASET --auxiliary-dataset RandomImages300k --name mvol_ensemble_seed"$seed" --method maxlogit --model-arch wideresnet 
done