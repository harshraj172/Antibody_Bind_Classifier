import argparse 
from run_experiments_BC import main
import optuna 

def objective(trial):
    config = dict.fromkeys(["optimizer_name", "learning_rate",
                            "num_layers", "hidden_dim", "temperature"])
    # Generate the optimizers.
    config["optimizer_name"] = trial.suggest_categorical("optimizer", ["adamW", "adam"])
    config["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # learning_rate_scheduler = trial.("learning_rate_scheduler": ["cosine", ""])
    config["num_layers"] = trial.suggest_int("num_layers", 6, 18)
    config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    config["temperature"] = trial.suggest_float("temperature", 0.1, 0.5, log=True)
    
    print(config)
    f1_score = main(
                    # data params
                    train_data_dir=args.train_data_dir,
                    test_data_dir=args.test_data_dir,
                    device1=args.device1,
                    device2=args.device2,
                    # trainer params
                    batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size,
                    learning_rate=config["learning_rate"],
                    epochs=args.epochs,
                    early_stopping=args.early_stopping,
                    # optimizer params
                    optimizer=config["optimizer_name"],
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
    parser.add_argument('--device1', default="cuda:0", type=str)
    parser.add_argument('--device2', default="cuda:0", type=str)
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=6, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--early_stopping', default="valid_loss", type=str)
    parser.add_argument('--weight_decay', default=7.459343285726558e-05, type=float)
    
    parser.add_argument('--use_struct', default=True, type=eval)
    parser.add_argument('--use_seq', default=False, type=eval)
    parser.add_argument('--pretrained_lm_model', default="Rostlab/prot_bert", type=str)
    parser.add_argument('--log_wandb', default=True, type=eval)
    
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
