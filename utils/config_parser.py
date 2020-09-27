import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model_id", help="Identifier for model")

# Data
parser.add_argument("-train_data", help="Train data", default="QQP_TRAIN", choices=["QQP_TRAIN", "PAWS_TRAIN"])
parser.add_argument("-dev_data", help="Dev data", default="QQP_DEV", choices=["QQP_TEST", "PAWS_TEST"])
parser.add_argument("-test_data", help="Test data", default="ontonotes/g_test.json")
parser.add_argument("-num_epoch", help="The number of epoch", default=5000, type=int)
parser.add_argument("-batch_size", help="The batch size", default=1000, type=int)
parser.add_argument("-eval_batch_size", help="The batch size", default=1998, type=int)
parser.add_argument("-goal", help="Limiting vocab to smaller vocabs (either ontonote or figer)", default="open",
                    choices=["open", "onto", "wiki", 'kb'])
parser.add_argument("-seed", help="Pytorch random Seed", default=1777)
parser.add_argument("-gpu", help="Using gpu or cpu", default=False, action="store_true")

# learning
parser.add_argument("-mode", help="Whether to train or test", default="train", choices=["train", "test", "dev"])
parser.add_argument("-learning_rate", help="start learning rate", default=0.001, type=float)
parser.add_argument("-mention_dropout", help="drop out rate for mention", default=0.5, type=float)
parser.add_argument("-input_dropout", help="drop out rate for sentence", default=0.2, type=float)

# Data ablation study
# parser.add_argument("-add_crowd", help="Add indomain data as train", default=False, action='store_true')
# parser.add_argument("-data_setup", help="Whether to use joint data set-up", default="single", choices=["single", "joint"])

# Model
#parser.add_argument("-multitask", help="Using a multitask loss term.", default=False, action='store_true')

# Save / log related
parser.add_argument("-model_save_dir", help="where to save model", default="_model", type=str)
#parser.add_argument("-eval_period", help="How often to run dev", default=500, type=int)
parser.add_argument("-log_dir", help="Where to save", default="log", type=str)

parser.add_argument("-load", help="Load existing model.", action='store_true')
parser.add_argument("-reload_model_name", help="")
