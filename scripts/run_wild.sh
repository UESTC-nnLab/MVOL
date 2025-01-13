IN_DATASET=CIFAR-10_subset

##########################
## Single Model Setting ##
##########################

### vanilla
for seed in 0 1000 2000 3000 4000
do
    echo "seed = $seed"
    python train_vanilla.py --exp-cat-name vanilla --model-arch wideresnet\
           --name vanilla_seed"$seed" --in-dataset ${IN_DATASET} --gpu 1 --droprate 0.0 --seed "$seed"\
           --depth 40 --width 2 --batch-size 128
    python eval_ood_detection.py --exp-cat-name vanilla --if-aux 0 --model-arch wideresnet\
           --epochs 100 --in-dataset ${IN_DATASET} --name vanilla_seed"$seed" --method maxlogit --gpu 1   
    python compute_metrics.py --exp-cat-name vanilla --if-aux 0 --model-arch wideresnet\
           --in-dataset ${IN_DATASET} --name vanilla_seed"$seed" --method maxlogit
done

### oe
for noise in 0 0.05 0.1 0.3 0.5
    for seed in 0 1000 2000 3000 4000 
    do
        echo "seed = $seed"
        python train_oe.py --exp-cat-name vanilla_oe --auxiliary-dataset mixed_cifar100 --model-arch wideresnet\
            --name oe_noise"$noise"_seed"$seed" --in-dataset ${IN_DATASET} --gpu $GPU --seed "$seed" --beta 0.5 --droprate 0.0\
            --batch-size 128 --ood-batch-size 128 --mixed-ritio 0.05
        python eval_ood_detection.py --exp-cat-name vanilla_oe --if-aux 1 --auxiliary-dataset mixed_cifar100 --model-arch wideresnet\
            --epochs 100 --in-dataset ${IN_DATASET} --name oe_noise"$noise"_seed"$seed" --method maxlogit --gpu $GPU  
        python compute_metrics.py --exp-cat-name vanilla_oe --if-aux 1 --auxiliary-dataset mixed_cifar100 --model-arch wideresnet\
            --in-dataset ${IN_DATASET} --name oe_noise"$noise"_seed"$seed" --method maxlogit
    done
done

### mvol
for noise in 0 0.05 0.1 0.3 0.5
    for seed in 0 1000 2000 3000 4000
    do
        echo "seed = $seed"
        python train_mvol.py --exp-cat-name mvol --auxiliary-dataset mixed_cifar100 --model-arch wideresnet\
                --name mvol_noise"$noise"_seed"$seed" --in-dataset ${IN_DATASET} --gpu $GPU --seed "$seed" --beta 0.5 --droprate 0.0\
                --batch-size 128 --ood-batch-size 128 --factor 1 --mixed-ritio 0.05
        python eval_ood_detection.py --exp-cat-name mvol --if-aux 1 --auxiliary-dataset mixed_cifar100 --model-arch wideresnet\
                --epochs 100 --in-dataset ${IN_DATASET} --name mvol_noise"$noise"_seed"$seed" --method maxlogit --gpu 0  
        python compute_metrics.py --exp-cat-name mvol --if-aux 1 --auxiliary-dataset mixed_cifar100 --model-arch wideresnet\
                --in-dataset ${IN_DATASET} --name mvol_noise"$noise"_seed"$seed" --method maxlogit
    done
done

########################################
## Ensemble Distilltion Model Setting ##
########################################

# vanilla
for seed in 0 1000 2000 3000 4000
do
    echo "seed = $seed"
    python train_vanilla_ensemble.py \
    --exp-cat-name vanilla_ensemble --name vanilla_ensemble_seed"$seed" --model-arch wideresnet\
    --teacher-names 'vanilla_seed0,vanilla_seed1000,vanilla_seed2000,vanilla_seed3000,vanilla_seed4000,vanilla_seed5000,vanilla_seed6000,vanilla_seed7000,vanilla_seed8000,vanilla_seed9000'\
    --in-dataset ${IN_DATASET} --gpu $GPU --seed 0 --alpha 0.5 --temperature 2 --depth 40 --width 2 --droprate 0.0 --batch-size 128
    python eval_ood_detection.py --exp-cat-name vanilla_ensemble --if-aux 0 --epochs 100 --in-dataset ${IN_DATASET} --name vanilla_ensemble_seed"$seed" --method maxlogit --gpu $GPU --model-arch wideresnet
    python compute_metrics.py --exp-cat-name vanilla_ensemble --if-aux 0 --in-dataset ${IN_DATASET} --name vanilla_ensemble_seed"$seed" --method maxlogit --model-arch wideresnet
done

# oe
for noise in 0 0.05 0.1 0.3 0.5
do
    for seed in 0 1000 2000 3000 4000
    do
        echo "seed = $seed"
        python train_oe_ensemble.py --exp-cat-name ensemble_oe --model-arch wideresnet\
            --name oe_ensemble_noise"$noise"_seed"$seed" \
            --in-dataset $IN_DATASET --auxiliary-dataset mixed_cifar100 --mixed-ritio $noise\
            --batch-size 128 --ood-batch-size 128 --gpu $GPU --seed $seed --alpha 0.5 --temperature 2 --beta 0.5 --depth 40 --width 2\
            --teacher-names 'vanilla_seed0,vanilla_seed1000,vanilla_seed2000,vanilla_seed3000,vanilla_seed4000,vanilla_seed5000,vanilla_seed6000,vanilla_seed7000,vanilla_seed8000,vanilla_seed9000'
        python eval_ood_detection.py --if-aux 1 --auxiliary-dataset mixed_cifar100 --exp-cat-name ensemble_oe --epochs 100 --in-dataset $IN_DATASET --model-arch wideresnet\
            --name oe_ensemble_noise"$noise"_seed"$seed" \
            --method maxlogit --gpu $GPU --depth 40 --width 2
        python compute_metrics.py --if-aux 1 --exp-cat-name ensemble_oe --in-dataset $IN_DATASET --auxiliary-dataset mixed_cifar100 --name oe_ensemble_noise"$noise"_seed"$seed" --method maxlogit --model-arch wideresnet 
    done
done

# mvol
for noise in 0 0.05 0.1 0.3 0.5
do
    for seed in 0 1000 2000 3000 4000
    do
        python train_mvol_ensemble.py --exp-cat-name mvol_ensemble --model-arch wideresnet\
            --name mvol_ensemble_noise"$noise"_seed"$seed" \
            --in-dataset $IN_DATASET --auxiliary-dataset mixed_cifar100 --mixed-ritio 0.1\
            --batch-size 128 --ood-batch-size 128 --gpu $GPU --seed $seed --alpha 0.5 --temperature 2 --beta 0.5 --depth 40 --width 2\
            --teacher-names 'vanilla_seed0,vanilla_seed1000,vanilla_seed2000,vanilla_seed3000,vanilla_seed4000,vanilla_seed5000,vanilla_seed6000,vanilla_seed7000,vanilla_seed8000,vanilla_seed9000'
        python eval_ood_detection.py --if-aux 1 --auxiliary-dataset mixed_cifar100 --exp-cat-name mvol_ensemble --epochs 100 --in-dataset $IN_DATASET --model-arch wideresnet\
            --name mvol_ensemble_noise"$noise"_seed"$seed" \
            --method maxlogit --gpu $GPU --depth 40 --width 2 
        python compute_metrics.py --if-aux 1 --exp-cat-name mvol_ensemble --in-dataset $IN_DATASET --auxiliary-dataset mixed_cifar100 --name mvol_ensemble_noise"$noise"_seed"$seed" --method maxlogit --model-arch wideresnet 
    done
done