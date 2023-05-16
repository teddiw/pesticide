import argparse

import numpy as np
import pandas as pd

from itertools import accumulate

def update_test(test_csv_file):
    """Combines disjointed test and labels csv files into one file."""
    test_set = pd.read_csv(test_csv_file)
    data_labels = pd.read_csv(test_csv_file[:-4] + "_labels.csv")
    for category in data_labels.columns[1:]:
        test_set[category] = data_labels[category]
    if "content" in test_set.columns:
        test_set.rename(columns={"content": "comment_text"}, inplace=True)
    test_set.to_csv(f"{test_csv_file.split('.csv')[0]}_updated.csv")
    return test_set


def create_val_set(csv_file, val_fraction):
    """Takes in a csv file path and creates a validation set
    out of it specified by val_fraction.
    """
    dataset = pd.read_csv(csv_file)
    np.random.seed(0)
    dataset_mod = dataset[dataset.toxic != -1]
    indices = np.random.rand(len(dataset_mod)) > val_fraction
    val_set = dataset_mod[~indices]
    val_set.to_csv("val.csv")

def expand_comment(comment):
    delimiter = " "
    temp = comment.split(delimiter)
    result = list(accumulate(temp, lambda x, y: delimiter.join([x, y])))
    return result
    
def convert_to_fudge_clf_data(csv_file, name):
    # can rename partial_df to df; it's all of the data, not a subset of the data
    partial_df = pd.read_csv(csv_file)
   
    partial_df.loc[:, 'comment_text'] = partial_df['comment_text'].apply(expand_comment)
    partial_df = partial_df.explode('comment_text')
    
    np.random.seed(0)
    indices = np.random.choice(np.arange(len(partial_df)), size=len(partial_df), replace=False)
    partial_df = partial_df.iloc[indices, :]

    partial_df.to_csv("all_fudge_toxicity_"+name+".csv")

def get_stats(csv_file):
    df = pd.read_csv(csv_file)
    print("SIZE:", df.size)
    print(df.head(5))
    # df['comment_text_length'] = df['comment_text'].str.len()
    # print('Average number of words in TOXIC comment:', np.mean(df[df['toxic']==1]['comment_text_length']))
    # print('Average number of words in NONTOXIC comment:', np.mean(df[df['toxic']==0]['comment_text_length']))
    
def make_small_dataset(csv_file, n, name):
    df = pd.read_csv(csv_file)
    small_df = df.iloc[:n, :]
    small_df.to_csv(name)

def limit_dataset(csv_file, max_length, name):
    df = pd.read_csv(csv_file)
    df = df.drop(df[df['comment_text'].str.len() >= max_length].index)
    df = df.dropna()
    df.to_csv(name)

def make_balanced_dataset(csv_file, name):
    df = pd.read_csv(csv_file)
    df_toxic = df[df['toxic']==1]
    df_nontoxic = df[df['toxic']==0]
    num_toxic = len(df_toxic)
    print('num_toxic', num_toxic)
    print('num_nontoxic', len(df_nontoxic))
    df_nontoxic = df_nontoxic.iloc[:num_toxic, :]
    new_df = pd.concat([df_toxic, df_nontoxic])
    print('num_new_df', len(new_df))
    np.random.seed(0)
    indices = np.random.choice(np.arange(len(new_df)), size=len(new_df), replace=False)
    new_df = new_df.iloc[indices, :]
    new_df.to_csv(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--val_csv", type=str)
    parser.add_argument("--data_for_fudge_csv", type=str)
    parser.add_argument("--name", type=str)

    parser.add_argument("--data_for_smaller_csv", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--max_length", type=int)

    parser.add_argument(
        "--update_test",
        action="store_true",
    )
    parser.add_argument(
        "--create_val_set",
        action="store_true",
    )
    parser.add_argument(
        "--create_fudge_data",
        action="store_true"
    )
    parser.add_argument(
        "--get_stats",
        action="store_true"
    )
    parser.add_argument(
        "--make_small_dataset",
        action="store_true",
    )
    parser.add_argument(
        "--limit_dataset",
        action="store_true",
    )
    parser.add_argument(
        "--make_balanced_dataset",
        action="store_true",
    )

    args = parser.parse_args()
    if args.update_test:
        test_set = update_test(args.test_csv)
    if args.create_val_set:
        create_val_set(args.val_csv, val_fraction=0.1)
    if args.create_fudge_data:
        convert_to_fudge_clf_data(args.data_for_fudge_csv, args.name)
    if args.get_stats:
        get_stats(args.data_for_fudge_csv)
    if args.make_small_dataset:
        make_small_dataset(args.data_for_smaller_csv, args.n, args.name)
    if args.limit_dataset:
        limit_dataset(args.data_for_smaller_csv, args.max_length, args.name)
    if args.make_balanced_dataset:
        make_balanced_dataset(args.data_for_smaller_csv, args.name)
