from argparse import ArgumentParser
from utils import dill_load, dill_dump, audio_length_per_model, won2020_default_config as config, path_models, get_model
import os
import torch
import random
import numpy as np

models = ["fcn", "sample", "musicnn"]
results = {}
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    

def get_counts_for_model(results):
    counts_for_model = {'top_k': 0, 'n_examples': 0, 'random': [], 'pos': [], 'k': []}
    counts_random = []
    counts_pos = []
    for item in results:
        label = results[item]['label']
        counts_for_model['top_k'] += label == results[item]['pred_top_k']
        counts_for_model['n_examples'] += 1
        counts_random.append(results[item]['pred_random'] == label)
        counts_pos.append(results[item]['pred_pos'] == label)
        counts_for_model['k'] = results[item]['k']
    counts_for_model['random'] = np.array(counts_random)
    counts_for_model['pos'] = np.array(counts_pos)
    return counts_for_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--n_random_samples", type=int, default=10)
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--model_type", type=str, choices=models)
    parser.add_argument("--analysis_path", type=str, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir
    n_random_samples = args.n_random_samples
    n_components = args.n_components
    m = args.model_type
    analysis_path = args.analysis_path

    mdir = os.path.join(input_dir, m)
    explanations = os.listdir(mdir) # only pt files
    explanations = [exp for exp in explanations if exp.endswith(".pt")]

    for exp_file in explanations:
        example_in = os.path.join(mdir, exp_file)
        exp = dill_load(example_in)

        label = list(exp.intercept.keys())[0] # works because we only explain one label

        top_components, component_indeces = exp.get_sorted_components(label,
                                                                      positive_components=True,
                                                                      negative_components=False,
                                                                      num_components='all',
                                                                      return_indeces=True)

        audio_length = audio_length_per_model[m]
        top_indeces = component_indeces[:n_components]
        last_pos_indeces = component_indeces[n_components:]
        print("last_pos_indeces", len(last_pos_indeces), last_pos_indeces)
        all_components = list(range(exp.factorization.get_number_components()))
        all_but_top_indeces = list(set(all_components).difference(top_indeces))

        config.model_type = args.model_type
        config.model_load_path = os.path.join(path_models, config.dataset, config.model_type, 'best_model.pth')
        config.input_length = audio_length_per_model[args.model_type]

        model = get_model(config)

        # prediction for top3 components
        x = torch.zeros(1, audio_length)
        selected_components = exp.factorization.compose_model_input(top_indeces)
        print("selected top components", top_indeces, selected_components.shape)
        x[0] = torch.tensor(selected_components).unsqueeze(0)
        x = x.to(device)
        pred_top_k = torch.argmax(model(x)).item()

        # prediction for random component
        x_random = torch.zeros(n_random_samples, audio_length)
        for i in range(n_random_samples):
            random_components = random.sample(all_but_top_indeces, n_components)
            selected_components = exp.factorization.compose_model_input(random_components)
            x_random[i] = torch.tensor(selected_components).unsqueeze(0)
        x_random = x_random.to(device)
        pred_random = torch.argmax(model(x_random), axis=1).detach().cpu().numpy()

        # prediction for random positive component
        try:
            x_pos = torch.zeros(n_random_samples, audio_length)
            for i in range(n_random_samples):
                random_components = random.sample(list(component_indeces[n_components:]), n_components)
                selected_components = exp.factorization.compose_model_input(random_components)
                x_pos[i] = torch.tensor(selected_components).unsqueeze(0)
            x_pos = x_pos.to(device)
            pred_pos = torch.argmax(model(x_pos), axis=1).detach().cpu().numpy()
        except:
            continue

        print(m, label, pred_top_k, pred_random, pred_pos)
        results[m + "_" + exp_file] = {"label": label, "pred_top_k":pred_top_k, "pred_random":pred_random, "pred_pos":pred_pos, "k":n_components}

    dill_dump(results, os.path.join(analysis_path, 'analyse_components_{}_{}.dl'.format(m, str(n_components).zfill(2))))

    counts_for_model = get_counts_for_model(results)

    print("{}/{} = {}".format(counts_for_model['top_k'], counts_for_model['n_examples'], counts_for_model['top_k'] / counts_for_model['n_examples']))

    print('random', counts_for_model['random'])
    print('pos', counts_for_model['pos'])

