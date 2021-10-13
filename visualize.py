from h5py import File as HDF5
from matplotlib import pyplot as plt
import numpy as np
from environment.utils import visualize_grasp
import sys
from filelock import FileLock
import os
import seaborn as sns
import pandas as pd
from utils import collect_stats
from tqdm import tqdm
import pickle
from pprint import pprint

def summarize(path):
    stats = collect_stats(path, int(1e7))
    for key, value in stats.items():
        if all(word not in key for word in ['distribution', 'img',
                                            'min', 'max', '_steps']):
            print(f'\t[{key:<36}]:\t{value:.04f}')
    # Episode lengths
    print('Easy Episode Lengths:')
    if 'episode_length/easy/distribution' in stats:
        easy_episode_lengths = stats['episode_length/easy/distribution']
        print('\tmean: {:.04f}'.format(
            np.mean(easy_episode_lengths)))
        print('\t25-quantile: {:.04f}'.format(
            np.quantile(easy_episode_lengths, 0.25)))
        print('\tmedian: {:.04f}'.format(
            np.median(easy_episode_lengths)))
        print('\t75-quantile: {:.04f}'.format(
            np.quantile(easy_episode_lengths, 0.75)))
    if 'episode_length/hard/distribution' in stats:
        hard_episode_lengths = stats['episode_length/hard/distribution']
        print('Hard Episode Lengths:')
        print('\tmean: {:.02f}'.format(
            np.mean(hard_episode_lengths)))
        print('\t25-quantile: {:.02f}'.format(
            np.quantile(hard_episode_lengths, 0.25)))
        print('\tmedian: {:.02f}'.format(
            np.median(hard_episode_lengths)))
        print('\t75-quantile: {:.02f}'.format(
            np.quantile(hard_episode_lengths, 0.75)))

    df = pd.DataFrame()
    averaged_coverages = []
    window = 10
    if False:
        final_coverage = stats['final_coverage/hard/distribution']
        final_coverage = stats['episode_delta_coverage/hard/distribution']
        temp = []
        for i in range(0, 150):
            if i + window > len(final_coverage):
                break
            temp.append(((i, i+window), final_coverage[i: i+window].mean()))
        temp.sort(key=lambda x: x[-1], reverse=True)
        pprint(temp)
    episodes = []
    best_coverage = stats['best_coverage/hard/distribution']
    for i in range(len(stats['best_coverage/hard/distribution'])):
        if i < window:
            continue
        for j in range(i-window, i+1):
            if j > 0:
                averaged_coverages.append(best_coverage[j])
                episodes.append(i)
    df['Final Coverage'] = averaged_coverages
    df['Episodes'] = episodes
    sns.lineplot(data=df, y='Final Coverage', x='Episodes')
    sns.despine()
    plt.title('Best Coverage over Training Episodes')
    plt.grid()
    plt.show()

    df = pd.DataFrame()
    averaged_coverages = []
    episodes = []
    final_coverage = stats['final_coverage/hard/distribution']
    for i in range(len(stats['final_coverage/hard/distribution'])):
        if i < window:
            continue
        for j in range(i-window, i+1):
            if j > 0:
                averaged_coverages.append(final_coverage[j])
                episodes.append(i)
    df['Final Coverage'] = averaged_coverages
    df['Episodes'] = episodes
    sns.lineplot(data=df, y='Final Coverage', x='Episodes')
    sns.despine()
    plt.title('Final Coverage over Training Episodes')
    plt.grid()
    plt.show()

    df = pd.DataFrame()
    delta_coverages = []
    difficulties = []
    steps = []
    for level in ['easy', 'hard']:
        for step, step_delta_coverages in \
                sorted(stats['delta_coverage_steps'][level].items(),
                       key=lambda x: int(x[0])):
            delta_coverages.extend(step_delta_coverages)
            steps.extend([step]*len(step_delta_coverages))
            difficulties.extend([level]*len(step_delta_coverages))
    df['Delta-Coverage'] = delta_coverages
    df['Difficulty'] = difficulties
    df['Episode Step'] = steps
    sns.lineplot(data=df, x='Episode Step',
                 y='Delta-Coverage', hue='Difficulty')
    sns.despine()
    plt.title('Delta-Coverage over Episode Steps')
    plt.grid()
    plt.show()

    # Post action coverages
    df = pd.DataFrame()
    postaction_coverages = []
    difficulties = []
    steps = []
    for level in ['easy', 'hard']:
        for step, step_postaction_coverages in \
                sorted(stats['postaction_coverage_steps'][level].items(),
                       key=lambda x: int(x[0])):
            postaction_coverages.extend(step_postaction_coverages)
            steps.extend([step]*len(step_postaction_coverages))
            difficulties.extend([level]*len(step_postaction_coverages))
    df['Postaction-Coverage'] = postaction_coverages
    df['Difficulty'] = difficulties
    df['Episode Step'] = steps
    sns.lineplot(data=df, x='Episode Step',
                 y='Postaction-Coverage', hue='Difficulty')
    sns.despine()
    plt.title('Postaction-Coverage over Episode Steps')
    plt.grid()
    plt.show()

    df = pd.DataFrame()
    action_primitive_proportions = []
    hues = []
    steps = []
    for level in ['easy', 'hard']:
        for step, count in stats['action_primitives_steps'][level].items():
            for action in count.keys():
                steps.append(step)
                action_primitive_proportions.append(count[action])
                hues.append(level + ' - ' + action)
    df['Action'] = action_primitive_proportions
    df['Difficulty'] = hues
    df['Episode Step'] = steps
    sns.lineplot(data=df, x='Episode Step', y='Action', hue='Difficulty')
    sns.despine()
    plt.title('Action Primitive Proportion Over Episode Steps')
    plt.grid()
    plt.show()


def simple_visualize(group, key, path_prefix, dir_path):
    fig = plt.figure()
    fig.set_figheight(3.2)
    fig.set_figwidth(13)
    gs = fig.add_gridspec(1, 5)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    img = np.array(group['pretransform_observations'])
    img = (np.swapaxes(img, 0, -1)*255).astype(np.uint8)
    ax.imshow(img[:, :, :3].astype(np.uint8))
    ax.set_title(' Coverage: {:.03f}'.format(
        group.attrs['preaction_coverage'] /
        group.attrs['max_coverage']))
    ax = fig.add_subplot(gs[0, 1:4])
    ax.axis('off')
    img = np.array(group['action_visualization']).astype(np.uint8)
    ax.imshow(img[:, :, :3])

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    img = np.array(group['next_observations'])
    img = (np.swapaxes(img, 0, -1)*255).astype(np.uint8)
    ax.imshow(img)
    ax.set_title(' Coverage: {:.03f}'.format(
        group.attrs['postaction_coverage'] /
        group.attrs['max_coverage']))

    output_path = path_prefix + '_before_after.png'
    plt.tight_layout(pad=0)
    plt.savefig(dir_path+output_path)
    plt.close(fig)
    return f'<td>{key} </td><td>' +\
        f'<img src="{output_path}" height="256px"> </td> '


if __name__ == "__main__":
    path = sys.argv[1]
    with FileLock(path + '.lock'):
        with HDF5(path, 'r') as file:
            keys = []
            for k in file.keys():
                try:
                    file[k].attrs['max_coverage']
                    keys.append(k)
                except:
                    pass
            print('keys:', len(keys))
    pprint(vars(pickle.load(
        open(path.split('replay_buffer.hdf5')[0] + 'args.pkl', 'rb'))))
    prefix = os.path.basename(os.path.dirname(path)) + '_'
    summarize(path)
    if input('visualize?') != 'y':
        exit()
    dir_path = os.path.dirname(path) + '/'
    webpage_path = dir_path + 'index.html'
    print(f'Outputing visualizations to {webpage_path}')

    with FileLock(path + '.lock'):
        with HDF5(path, 'r') as file:
            use_simple_vis = 'all_obs' not in file[keys[0]]\
                or 'action_visualization' not in file[keys[0]]
            visualization_fn = simple_visualize\
                if use_simple_vis \
                else visualize_grasp
            output = """
    <style>
        table,
        th,
        td {
            border: 1px solid black;
            border-collapse: collapse;
        }

        .slidecontainer {
            width: 100%;
            /* Width of the outside container */
        }

        /* The slider itself */
        .slider {
            -webkit-appearance: none;
            /* Override default CSS styles */
            appearance: none;
            width: 100%;
            /* Full-width */
            height: 25px;
            /* Specified height */
            background: #d3d3d3;
            /* Grey background */
            outline: none;
            /* Remove outline */
            opacity: 0.7;
            /* Set transparency (for mouse-over effects on hover) */
            -webkit-transition: .2s;
            /* 0.2 seconds transition on hover */
            transition: opacity .2s;
        }

        /* Mouse-over effects */
        .slider:hover {
            opacity: 1;
            /* Fully shown on mouse-over */
        }
    </style>
    <div class="slidecontainer">
        <p>Speed</p>
        <input type="range" min="1" max="10" value="5" class="slider" id="myRange">
    </div>
            """
            script = """
            <script>
                let slider = document.getElementById("myRange");

                function updateVideoSpeed(speed) {
                    let vids = document.getElementsByTagName('video')
                    // vids is an HTMLCollection
                    for (let i = 0; i < vids.length; i++) {
                        //#t=0.1
                        vids.item(i).playbackRate = speed;
                    }
                }
                updateVideoSpeed(slider.value)

                // Update the current slider value (each time you drag the slider handle)
                slider.oninput = function () {
                    updateVideoSpeed(this.value)
                }

            </script>
            """
            output += '<table style="width:100%">'
            for k in tqdm(keys):
                output += '<tr>'
                group = file.get(k)
                output += visualization_fn(
                    group=group,
                    key=k,
                    path_prefix=prefix + k,
                    dir_path=dir_path)
                output += '</tr>'
                with open(webpage_path, 'w') as webpage:
                    webpage.write(output + '</table>' + script)
