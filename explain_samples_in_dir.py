import os
import torch
from torch.autograd import Variable

from audioLIME.factorization import SpleeterFactorization
from audioLIME import lime_audio
from audioLIME.data_provider import RawAudioProvider

from utils import dill_load, dill_dump, composition_fn, create_predict_fn, audio_length_per_model, \
    np_to_audio_file, prepare_audio, won2020_default_config as config, tags_msd as tags, path_models

from training.eval import Predict

import matplotlib.pyplot as plt
import librosa.display

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="fcn")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_temporal_segments", type=int)
    parser.add_argument("--n_display_components", type=int, default=3)
    parser.add_argument("--n_chunks", type=int, default=16)
    parser.add_argument("--use_global_tag", action='store_true', default=False)

    args = parser.parse_args()

    print(args)

    path_experiments = args.out_dir
    n_display_components = args.n_display_components
    batch_size = args.batch_size
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = 'exhaustive'
    n_segments = args.n_temporal_segments
    samples_path = args.samples_dir
    available_samples = os.listdir(args.samples_dir)

    sample_rate = 16000
    overwrite = False

    config.model_type = args.model_type
    config.model_load_path = os.path.join(path_models, config.dataset, config.model_type, 'best_model.pth')
    config.input_length = audio_length_per_model[args.model_type]
    config.batch_size = args.n_chunks

    model = get_model(config)

    for sample in available_samples:
        audio_path = os.path.join(samples_path, sample)
        data_provider = RawAudioProvider(audio_path)
        x, snippet_starts = prepare_audio(audio_path, config.input_length, nr_chunks=config.batch_size, return_snippet_starts=True)

        print(snippet_starts)
        x = x.cuda()
        x = Variable(x)
        outputs = model(x)
        top_tag_per_snippet = torch.argmax(outputs.detach().cpu(), axis=1)
        print("top_tag_per_snippet", len(top_tag_per_snippet), top_tag_per_snippet)

        sorted_args = torch.argsort(outputs.detach().cpu().mean(axis=0), descending=True)

        print([tags[t] for t in sorted_args[0:3]])

        top_idx = sorted_args[0].item()
        sorted_snippets = torch.argsort(outputs[:, top_idx].detach().cpu()).numpy()
        print("top idx:", top_idx)
        print("top segments:", sorted_snippets)

        predict_fn = create_predict_fn(model, config)
        spleeter_factorization = SpleeterFactorization(data_provider,
                                                       n_temporal_segments=n_segments,
                                                       composition_fn=composition_fn,
                                                       model_name='spleeter:5stems')

        explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

        for sn in range(len(snippet_starts)):

            if args.use_global_tag:
                labels = [top_idx]
            else:
                snippet_tag = top_tag_per_snippet[sn].item()
                labels = [snippet_tag]

            print("processing {}_{}".format(sample, sn))

            explanation_name = "{}/{}_cls{}_sntag{}_nc{}_sn{}_seg{}_smp{}_nd{}".format(config.model_type, sample, top_idx, labels[0], config.batch_size, sn, n_segments, num_samples, n_display_components)
            explanation_path = os.path.join(path_experiments, explanation_name + ".pt")

            spleeter_factorization.set_analysis_window(snippet_starts[sn], config.input_length)
            print("mix length", len(spleeter_factorization.data_provider.get_mix()))

            if os.path.exists(explanation_path) and not overwrite:
                print("Loading explanation ...")
                explanation = dill_load(explanation_path)
            else:
                print("(Re)computing explanation ...")
                explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                                         predict_fn=predict_fn,
                                                         labels=labels,
                                                         num_samples=num_samples,
                                                         batch_size=batch_size
                                                         )
                dill_dump(explanation, explanation_path)

            print(explanation.local_exp)

            top_components, component_indeces = explanation.get_sorted_components(labels[0],
                                                                                  positive_components=True,
                                                                                  negative_components=False,
                                                                                  num_components=n_display_components,
                                                                                  return_indeces=True)
            if len(top_components) == 0:
                print("No positive components found")
                continue
            if len(top_components) > 1:
                summed_components = sum(top_components)
                fig = plt.figure()
                for comp in range(len(top_components)):
                    plt.subplot(len(top_components), 1, comp + 1)
                    librosa.display.waveplot(top_components[comp], sr=sample_rate)
                plt.savefig(explanation_path.replace(".pt", "fig1.png"))
            else:
                summed_components = top_components[0]
                fig = plt.figure()
                librosa.display.waveplot(summed_components, sr=sample_rate)
                plt.savefig(explanation_path.replace(".pt", "fig1.png"))

            librosa.output.write_wav(explanation_path.replace(".pt", ".wav"), summed_components, sample_rate)
            librosa.output.write_wav(explanation_path.replace(".pt", "_original.wav"), spleeter_factorization.data_provider.get_mix(), sample_rate)

