■ Requirement
    To set up their environment, please run:
    (we recommend to use Anaconda for installation.)
    $ conda env create -n LML -f LML.yml
    $ conda activate LML

■ Dataset
    You can create dataset by running following code. Dataset will be saved in ./data directory.
    $ python ./script_supplementary/toy_exp_script/makebag_script/crossvali_make_dataset_10class_uniform.py

■ Step1: Training the Counting Network. The trained model is used as a pre-trained model in MPEM in the next step.
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.1 --module "Count" --temper1 0.1 --temper2 0.1 --dataset "cifar10" --classes 10 --is_evaluation 0

■ Step2: Training the model using bags in which the majority proportion has been enhanced by MPEM, with the removal ratio 'r' ranging from 0.1 to 1.0.
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.1 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.2 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.3 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.4 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.5 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.6 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.7 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.8 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 0.9 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0
    $ python ./script_supplementary/main.py --non_pos_mask_rate 1.0 --module "MPEM" --dataset "cifar10" --classes 10 --is_evaluation 0

■ Step3: Selecting the optimal removal ratio based on the validation loss, and performing inference on the test data using the model trained with the selected ratio.
    $ python ./script_supplementary/select_optimal_k_main.py --module "MPEM" --dataset "cifar10" --classes 10 
