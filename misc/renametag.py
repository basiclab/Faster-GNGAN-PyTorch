from pathlib import Path

import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event
from tqdm import tqdm


MAP = {
    'FID_EMA': 'FID/EMA',
    'IS_EMA': 'IS/EMA',
    'IS_std_EMA': 'IS/EMA/std',
    'IS_std': 'IS/std',
    'loss_cr': 'loss/cr',
    'loss_real': 'loss/real',
    'loss_fake': 'loss/fake',
}


def rename_events(input_path, output_path):
    # Make a record writer
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([str(input_path)]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    # Check if the tag should be renamed
                    if v.tag in MAP:
                        # Rename with new tag name
                        v.tag = MAP[v.tag]
            writer.write(ev.SerializeToString())


def rename_events_dir(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Make output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # Iterate event files
    with tqdm(list(input_dir.glob('**/*.tfevents*'))) as pbar:
        for ev_file in pbar:
            # Make directory for output event file
            if '2xlr' not in str(ev_file.parent):
                out_file = Path(output_dir, ev_file.relative_to(input_dir))
                out_file.parent.mkdir(parents=True, exist_ok=True)
                pbar.write(out_file.stem)
                # Write renamed events
                rename_events(ev_file, out_file)


if __name__ == '__main__':
    rename_events_dir('/nfs/home/yilun/tmp_logs', '/nfs/home/yilun/logs')
