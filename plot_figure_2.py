from argparse import ArgumentParser
from visualize_component_analysis import get_counts_for_model, get_xy_from_results
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--analysis_path_audiolime", type=str, required=True)
    parser.add_argument("--analysis_path_soundlime", type=str, required=True)
    parser.add_argument("--figure_path", type=str, default='/share/home/verena/experiments/understanding_musictagging/')
    parser.add_argument("--font_size", type=int, default=14)
    parser.add_argument("--theme", default="seaborn")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=4)
    args = parser.parse_args()

    print(args)

    fontsize_general = args.font_size
    fontsize_ticks = fontsize_general
    fontsize_axes = fontsize_general + 2
    fontsize_title = fontsize_general + 4

    models = {'sample': 'SampleCNN', 'fcn': 'FCN'}

    plt.style.use(args.theme)
    # plt.rcParams.update({'font.size': args.font_size})

    f, axes = plt.subplots(1, 2, sharey=True, figsize=(args.width, args.height))
    for i, model in enumerate(models):

        analysis_path_a = args.analysis_path_audiolime
        analysis_path_s = args.analysis_path_soundlime

        results_a = sorted([resfile for resfile in os.listdir(analysis_path_a) if model in resfile and resfile.endswith(".dl")])
        results_s = sorted([resfile for resfile in os.listdir(analysis_path_s) if model in resfile and resfile.endswith(".dl")])

        x_a, y_top_a, y_pos_a, _ = get_xy_from_results(analysis_path_a, results_a)
        x_s, y_top_s, y_pos_s, _ = get_xy_from_results(analysis_path_s, results_s)

        # ax = plt.subplot(1, 2, i+1)
        # if i == 0:
        #     ax1 = ax
        axes[i].plot(x_a, y_top_a, "o--", label="audioLIME top-k")
        axes[i].plot(x_s, y_top_s, "o--", label="SLIME top-k")

        axes[i].errorbar(x_a.squeeze(), y=y_pos_a.mean(axis=1), yerr=y_pos_a.std(axis=1), fmt='o--', label='audioLIME random')
        # plt.errorbar(x_s.squeeze(), y=y_pos_s.mean(axis=1), yerr=y_pos_s.std(axis=1), fmt='--', label='SoundLIME positive')

        axes[i].set_title(models[model], fontsize=fontsize_title)
        axes[i].set_xticks(x_a)
        axes[i].set_xticklabels([str(int(x)) for x in x_a], fontsize=fontsize_ticks)
        axes[i].set_xlabel("$k$ interpretable components", fontsize=fontsize_axes)
        axes[i].yaxis.grid(True)
        axes[i].xaxis.grid(False)


        # plt.setp(axes[i].get_xticklabels(), fontsize=fontsize_ticks)
        # plt.setp(axes[i].get_xticks(), visible=True)

    axes[0].legend(loc='upper left', fontsize=fontsize_general)
    axes[0].set_ylabel("% same top tag", fontsize=fontsize_axes)
    plt.setp(axes[0].get_yticklabels(), fontsize=fontsize_ticks)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figure_path, "audiolime_slime.png"))
    plt.show()