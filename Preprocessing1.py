from pypianoroll import Multitrack, Track
import os, pretty_midi
import numpy as np

raw_midi_dir = '/Users/apple/Downloads/Pop/'
merged_midi_dir = '/Users/apple/Downloads/Pop_merged_midi/'
filtered_midi_dir = '/Users/apple/Downloads/Pop_filtered_midi/'
raw_npy_dir = '/Users/apple/Downloads/Pop_npy/'


def get_merged(multitrack):
    """分类并合并乐器轨道，吉他、贝斯、钢琴，其余全部并入弦乐"""
    category_list = {'drum': [],'piano': []}
    program_dict = {'piano': 0, 'drum': 0}

    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            category_list['drum'].append(idx)
        else: #track.program//8 == 0:
            category_list['piano'].append(idx)
        # elif track.program//8 == 3:
        #     category_list['Guitar'].append(idx)
        # elif track.program//8 == 4:
        #     category_list['Bass'].append(idx)
        # else:
        #     category_list['Strings'].append(idx)

    tracks = []
    if category_list['drum']:
        drum_merged = multitrack[category_list['drum']].get_merged_pianoroll()
        tracks.append(Track(drum_merged, program_dict['drum'], True, 'drum'))

    if category_list['piano']:
        piano_merged = multitrack[category_list['piano']].get_merged_pianoroll()
        tracks.append(Track(piano_merged, program_dict['piano'], False, 'piano'))

    # for key in category_list:
    #     if category_list[key]:
    #         merged = multitrack[category_list[key]].get_merged_pianoroll()
    #         tracks.append(Track(merged, program_dict[key], merged.is_drum, key))
    #     else:
    #         pass
            #tracks.append(Track(None, program_dict[key], False, key))
    merged = Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat, multitrack.beat_resolution, multitrack.name)
    #merged.save(npz_dir + multitrack.name + '.npz')
    return merged

def get_midi_info(file):
    """通过PrettyMIDI对象作为媒介，获取midi文件的信息，用以筛选"""
    pm = pretty_midi.PrettyMIDI(file)
    
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time # 第一种TimeSignature的开始时间，秒
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes() # when tempo changes(in seconds), tempo at these times; both lists

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None,
        'num_instruments': len(pm.instruments)
    }

    return midi_info

def midi_filter(midi_info):
    """filter midi files, return True for qualified midis, return False for others"""
    if midi_info['first_beat_time'] > 0.0:
        #print(midi_info['first_beat_time'])
        return False
    elif midi_info['num_time_signature_change'] > 1:
        #print(midi_info['num_time_signature_change'])
        return False
    elif midi_info['time_signature'] not in ['4/4']:
        #print(midi_info['time_signature'])
        return False
    elif midi_info['num_instruments'] < 2:
        return False
    else:
        return True

def get_midi_file(root):
    """Return a list of MIDI files in `root` (recursively)"""
    files = []
    for _, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mid'):
                files.append(filename)
    return files


def get_bar_piano_roll(piano_roll, last_bar_mode='remove'):
    if int(piano_roll.shape[0] % 64) is not 0:
        if last_bar_mode == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif last_bar_mode == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128, 2)
    return piano_roll

# MAIN FUNCTION
# make sure the directories exist
if not os.path.exists(raw_midi_dir):
    os.makedirs(raw_midi_dir)

if not os.path.exists(merged_midi_dir):
    os.makedirs(merged_midi_dir)

if not os.path.exists(filtered_midi_dir):
    os.makedirs(filtered_midi_dir)

if not os.path.exists(raw_npy_dir):
    os.makedirs(raw_npy_dir)

midi_files = get_midi_file(raw_midi_dir) # returns a list of all midi file names in 'raw_midi_dir'
print(midi_files) # print that list

for file in midi_files:
    try:
        midi_info_1 = get_midi_info(os.path.join(raw_midi_dir, file))
    except:
        continue
    if midi_filter(midi_info_1):
        try:
            multitrack = Multitrack(os.path.join(raw_midi_dir, file))
        except:
            continue
        multitrack.write(os.path.join(filtered_midi_dir, file))
        print(file+" filtered")
        merged = get_merged(multitrack) # merge the tracks played by the same instruments
        merged.write(os.path.join(merged_midi_dir, file)) # save merged midi files
        midi_info_2 = get_midi_info(os.path.join(merged_midi_dir, file))
        if midi_filter(midi_info_2):
    #midi_info = get_midi_info(os.path.join(merged_midi_dir, file))
    #print(file)
    #print(midi_filter(midi_info))
    #if midi_filter(midi_info):
    #    merged.write(os.path.join(filtered_midi_dir, file)) # save filtered midi files
        #merged.save(npz_dir + file.split('.')[0] + '.npz')
            stacked = merged.get_stacked_pianoroll() # returns ndarray, a multi-track pianoroll
            print(stacked.shape) # (7824, 128, 2)

            pr = get_bar_piano_roll(stacked)
            print(pr.shape)
            pr_clip = pr[:, :, 24:108, :] # 将第三个维度切断至24-108，长度变为84
            print(pr_clip.shape)
        #pr_re = pr_clip.reshape(-1, 64, 84, 2)
        #print(pr_re.shape)
            np.save(os.path.join(raw_npy_dir, os.path.splitext(file)[0] + '.npy'), pr_clip) # save 4-dim npy files
