def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, seq_emb, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()
  
    num_iters = len(dataloader)
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        hAB, hAG = torch.tensor([0]), torch.tensor([0])
        if FLAGS.use_seq:
            hAB, hAG = seq_emb.pretrained_emb(seqAB), seq_emb.pretrained_emb(seqAG)
        hAB = hAB.to(FLAGS.device)
        hAG = hAG.to(FLAGS.device)
        gAB = gAB.to(FLAGS.device)
        gAG = gAG.to(FLAGS.device)
        y = y.to(FLAGS.device)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(gAB, hAB, gAG, hAG)
        l1_loss, __, rescale_loss = loss_fnc(pred, y)

        # backprop
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} rescale loss: {rescale_loss:.5f} [units]")
        if FLAGS.use_wandb:
            if i % FLAGS.log_interval == 0:
                wandb.log({"Train L1 loss": to_np(l1_loss), 
                        "Rescale loss": to_np(rescale_loss)})

        if FLAGS.profile and i == 10:
            sys.exit()

        scheduler.step(epoch + i / num_iters)

def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        gAB = gAB.to(FLAGS.device)
        gAG = gAG.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(gAB, gAG).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
        
        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
    rloss /= FLAGS.val_size
    results_df = metric(Y_true.reshape(-1), Y_pred.reshape(-1))
    
    print(f"...[{epoch}|val] rescale loss: {rloss:.5f} [units]")
    print(results_df)
    if FLAGS.use_wandb:
        wandb.log({"val_precision": result_df['Precision'][0],
                   "val_recall": result_df['Recall'][0], 
                   "val_F1_score": result_df['F1 Score'][0], 
                   "val_L1_loss": to_np(rloss)})

def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        gAB = gAB.to(FLAGS.device)
        gAG = gAG.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(gAB, gAG).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
        
        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
    rloss /= FLAGS.test_size
    results_df = metric(Y_true.reshape(-1), Y_pred.reshape(-1))
    
    print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")
    print(results_df)
    if FLAGS.use_wandb:
        wandb.log({"test_precision": result_df['Precision'][0],
                   "test_recall": result_df['Recall'][0], 
                   "test_F1_Score": result_df['F1 Score'][0], 
                   "test_L1_loss": to_np(rloss)})


def collate(samples):
    structseqAB_lst, structseqAG_lst, y = map(list, zip(*samples))
    batched_graphAB = dgl.batch([s['struct'] for s in structseqAB_lst])
    batched_graphAG = dgl.batch([s['struct'] for s in structseqAG_lst])
    seqAB_lst = [('protein', s['seq'][:1017]) for s in structseqAB_lst]
    seqAG_lst = [('protein', s['seq'][:1017]) for s in structseqAG_lst]
    return batched_graphAB, seqAB_lst, batched_graphAG, seqAG_lst, torch.tensor(y)


def main(FLAGS, UNPARSED_ARGV):

    # Prepare data
    train_dataset = _Antibody_Antigen_Dataset_(FLAGS.train_data[0], FLAGS.train_data[1])
    print("Dataset Created!!")
    train_loader = DataLoader(train_dataset, 
                              batch_size=FLAGS.batch_size, 
                              shuffle=True, 
                              collate_fn=collate, 
                              num_workers=FLAGS.num_workers)

#     val_dataset = _Antibody_Antigen_Dataset_(FLAGS.val_data[0], FLAGS.val_data[1]) 
#     val_loader = DataLoader(val_dataset, 
#                             batch_size=FLAGS.batch_size, 
#                             shuffle=False, 
#                             collate_fn=collate, 
#                             num_workers=FLAGS.num_workers)

#     test_dataset = _Antibody_Antigen_Dataset_(FLAGS.test_data[0], FLAGS.test_data[1]) 
#     test_loader = DataLoader(test_dataset, 
#                              batch_size=FLAGS.batch_size, 
#                              shuffle=False, 
#                              collate_fn=collate, 
#                              num_workers=FLAGS.num_workers)

    FLAGS.train_size = len(train_dataset)
#     FLAGS.val_size = len(val_dataset)
#     FLAGS.test_size = len(test_dataset)

    # Choose model
    seq_emb = models.get_SeqEmb(FLAGS.pretrained_lm_model)
    model = models.StructSeqNet(FLAGS.use_struct, 
                         FLAGS.use_seq,
                         FLAGS.num_layers, 
                         train_dataset.node_feature_size, 
                         train_dataset.edge_feature_size,
                         FLAGS.pretrained_lm_model, 
                         FLAGS.pretrained_lm_emb_dim,
                         num_channels=FLAGS.num_channels,
                         num_nlayers=FLAGS.num_nlayers,
                         num_degrees=FLAGS.num_degrees,
                         div=FLAGS.div,
                         pooling=FLAGS.pooling,
                         n_heads=FLAGS.head)
    # model = StructNet(FLAGS.num_layers, 
    #                   train_dataset.node_feature_size, 
    #                   train_dataset.edge_feature_size,
    #                   num_channels=FLAGS.num_channels,
    #                   num_nlayers=FLAGS.num_nlayers,
    #                   num_degrees=FLAGS.num_degrees,
    #                   div=FLAGS.div,
    #                   pooling=FLAGS.pooling,
    #                   n_heads=FLAGS.head)


    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)
    #wandb.watch(model)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               FLAGS.num_epochs, 
                                                               eta_min=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')
    
    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, seq_emb, task_loss, train_loader, optimizer, scheduler, FLAGS)
#         val_epoch(epoch, model, seq_emb, task_loss, val_loader, FLAGS)
#         test_epoch(epoch, model, seq_emb, task_loss, test_loader, FLAGS)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()\
    
    parser.add_argument('--use_seq', type=eval, default=False,
            help="Use sequence info of protein")
    parser.add_argument('--use_struct', type=eval, default=True,
            help="Use structure info of protein")
    
    # Model parameters
    parser.add_argument('--model', type=str, default='SeqNet', 
            help="String name of model")
    parser.add_argument('--pretrained_lm_model', type=str, default='esm1b_t33_650M_UR50S',
            help="Pretrained LM model name")
    parser.add_argument('--pretrained_lm_emb_dim', type=int, default=1280,
            help="Pretrained LM model out dim")
    
    parser.add_argument('--num_layers', type=int, default=1,
            help="Number of equivariant layers") #4
    parser.add_argument('--num_degrees', type=int, default=2,
            help="Number of irreps {0,1,...,num_degrees-1}") #4
    parser.add_argument('--num_channels', type=int, default=4,
            help="Number of channels in middle layers") #16
    parser.add_argument('--num_nlayers', type=int, default=0,
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4,
            help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg',
            help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1,
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=1, 
            help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, 
            help="Number of epochs")

    # Data
    parser.add_argument('--meta_data_address', type=str, default='data/SabDab/sample_sabdab_summary.csv',
            help="Address to structure file")
    parser.add_argument('--train_data', type=list, default=['data/SabDab/X_Ab.json', 'data/SabDab/X_Ag.json'],
            help="training data - Antibodies, Antigens")
    parser.add_argument('--val_data', type=list, default=['data/SabDab/X_Ab.json', 'data/SabDab/X_Ag.json'],
            help="training data - Antibodies, Antigens")
    parser.add_argument('--test_data', type=list, default=['data/SabDab/X_Ab.json', 'data/SabDab/X_Ag.json'],
            help="training data - Antibodies, Antigens")


    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--use_wandb', type=eval, default=False,
            help="To use wandb or not - [True, False]")
    parser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
            help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
            help="Path to model to restore")
    parser.add_argument('--wandb', type=str, default='equivariant-attention', 
            help="wandb project name")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=0, 
            help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
            help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{FLAGS.num_degrees}-l{FLAGS.num_layers}-{FLAGS.num_channels}-{FLAGS.num_nlayers}'

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if FLAGS.use_wandb:
        # Log all args to wandb
        if FLAGS.name:
            wandb.init(project=f'{FLAGS.wandb}', name=f'{FLAGS.name}')
        else:
            wandb.init(project=f'{FLAGS.wandb}')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    main(FLAGS, UNPARSED_ARGV)
