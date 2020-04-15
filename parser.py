import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run wsdm project")
    # global
    parser.add_argument("--port",
                        type=str,
                        default="6666",
                        help="port num")
    parser.add_argument("--cuda_devices",
                        type=int,
                        nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="cuda devices")
    parser.add_argument("--mode",
                        type=str,
                        choices=["train", "eval"],
                        help="train or eval")
    parser.add_argument("--re_train",
                        action="store_true",
                        help="retrain the model")
    parser.add_argument("--num_worker",
                        type=int,
                        default=4,
                        help="number of data loader worker")
    parser.add_argument("--max_norm",
                        type=int,
                        default=30,
                        help="max norm value")
    parser.add_argument("--t_max",
                        type=int,
                        default=20,
                        help="CosineAnnealingLR. Maximum number of iterations")
    parser.add_argument("--eta_min",
                        type=float,
                        default=1e-9,
                        help="CosineAnnealingLR. Minimum learning rate.")
    parser.add_argument("--data",
                        type=str,
                        default="./data/train.csv",
                        help="data path")
    parser.add_argument("--model_path",
                        type=str,
                        default="./checkpoints/model.pth",
                        help="model file path")
    parser.add_argument("--train_model",
                        type=str,
                        default="./checkpoints/model.pth",
                        help="model file path")
    # train
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--epoch",
                        type=int,
                        default=108,
                        help="num epoch")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("--loss_margin",
                        type=float,
                        default=0.6,
                        help="loss margin, default is 0.6. if margin is None, it uses auto margin")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="god knows seed")
    parser.add_argument("--embedding_dim",
                        type=int,
                        default=300,
                        help="embedding dimenstion")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=300,
                        help="hidden dimenstion")
    parser.add_argument("--vocab_size",
                        type=int,
                        default=400000,
                        help="vocab size")
    parser.add_argument("--target",
                        type=str,
                        default="description_text",
                        help="description_text")
    parser.add_argument("--source",
                        type=str,
                        default="title_abstract",
                        help="title + abstract")
    return parser.parse_args()
