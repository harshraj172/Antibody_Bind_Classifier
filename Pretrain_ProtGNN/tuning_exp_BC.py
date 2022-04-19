import argparse 
from run_experiments_BC import main
import optuna 

def objective(trial):
    config = dict.fromkeys(["learning_rate", "num_layers", "hidden_dim", "temperature"])
    
    # Generate the optimizers.
    lr_range = [float(k) for k in args.lr_range.strip('][').split(', ')]
    config["learning_rate"] = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
    print(config["learning_rate"])
    
    num_layers_range = [int(k) for k in args.num_layers_range.strip('][').split(', ')]
    config["num_layers"] = trial.suggest_int("num_layers", num_layers_range[0], num_layers_range[1])
    
    hidden_dim_lst = [int(k) for k in args.hidden_dim_lst.strip('][').split(', ')]
    config["hidden_dim"] = trial.suggest_categorical("hidden_dim", hidden_dim_lst)
    
    temp_range = [float(k) for k in args.temp_range.strip('][').split(', ')]
    config["temperature"] = trial.suggest_float("temperature", temp_range[0], temp_range[1], log=True)
    
    print(config)
    
    f1_score = main(
                    # data params
                    train_data_dir=args.train_data_dir,
                    test_data_dir=args.test_data_dir,
                    device_ids=args.device_ids,
                    # trainer params
                    batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size,
                    learning_rate=config["learning_rate"],
                    epochs=args.epochs,
                    early_stopping=args.early_stopping,
                    # optimizer params
                    weight_decay=args.weight_decay,
                    # model params
                    use_struct=args.use_struct,
                    use_seq=args.use_seq,
                    pretrained_lm_model=args.pretrained_lm_model,
                    hidden_dim=config["hidden_dim"],
                    num_layers=config["num_layers"],
                    temperature=config["temperature"],
                    # experiment params
                    log_wandb=args.log_wandb,
                    )
    return f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Binary Classifer')

    parser.add_argument('--train_data_dir', default="data/SabDab/train", type=str)
    parser.add_argument('--test_data_dir', default="data/SabDab/test", type=str)
    parser.add_argument('--device_ids', default="[0]", type=str)
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=6, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--early_stopping', default="valid_loss", type=str)
    parser.add_argument('--weight_decay', default=7.459343285726558e-05, type=float)
    
    #params to tune
    parser.add_argument('--lr_range', default="[1e-5, 1e-1]", type=int)
    parser.add_argument('--num_layers_range', default="[6, 12]", type=int)
    parser.add_argument('--hidden_dim_lst', default="[64, 128, 256]", type=str)
    parser.add_argument('--temp_range', default="[0, 0.5]", type=float)
    
    parser.add_argument('--use_struct', default=True, type=eval)
    parser.add_argument('--use_seq', default=False, type=eval)
    parser.add_argument('--pretrained_lm_model', default="Rostlab/prot_bert", type=str)
    parser.add_argument('--log_wandb', default=True, type=eval)
    
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
