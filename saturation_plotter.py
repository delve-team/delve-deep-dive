from matplotlib import pyplot as plt
import pathlib
import pandas as pd


def order_columns(column_names):
    column_names = sorted(column_names,
                          key=lambda x: int(x.split('-')[-1]) + 100 if 'classifier' in x else int(x.split('-')[-1]))
    column_names.remove
    return column_names


def extract_layer_saturation(df, excluded='classifier-6', epoch=19):
    #df = df.reindex(sorted(df.columns), axis=1)
    cols = list(df.columns)
    # print(cols)
    train_cols = [col for col in cols if
                  'train' in col and not excluded in col and not 'accuracy' in col and not 'loss' in col]
#    train_cols = order_columns(train_cols)

    # print(list(train_cols))
    epoch_df = df[df.index.values == epoch]

    accuray = epoch_df['test_accuracy'].values[0]
    epoch_df = epoch_df[train_cols]
    #epoch_df.reindex(sorted(df.columns), axis=1)

    return epoch_df, accuray


def plot_saturation_level(df, acc=-1, savepath='run.png', epoch=0):
    plt.clf()
    ax = plt.gca()
    cols = list(df.columns)
    col_names = [i for i in df.columns]
   # ax.grid()
    ax.bar(list(range(len(col_names))), df.values[0])
    plt.xticks(list(range(len(col_names))), [col_name.replace('train-saturation_', '') for col_name in col_names], rotation=90)
    ax.set_ylim((0,100))
    ax.text(1, 80, 'Accuray: {}'.format(acc))
    plt.yticks(fontsize=16)
    plt.xlabel('Layers', fontsize=16)
    plt.title(pathlib.Path(savepath).name.replace('_', ' ').replace('.csv', f' epoch: {epoch}'), fontsize=16)
    plt.ylabel('Saturation in %', rotation='vertical', fontsize=16)
    plt.tight_layout()
    plt.savefig(savepath.replace('.csv', f'_epoch_{epoch}.png'))
    return


def plot_saturation_level_from_results(savepath, epoch):
    print('Plotting Saturation')
    df = pd.read_csv(savepath, sep=';')
    epoch_df, acc = extract_layer_saturation(df, epoch=epoch)
    plot_saturation_level(epoch_df, acc, savepath, epoch)
    print("Plot saved")
