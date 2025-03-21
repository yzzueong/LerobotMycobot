"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.1 to
2.0.
Usage:

```bash
python .\lerobot\common\datasets\v2\convert_data_v21_to_v20.py
  --repo-id=xxx/xxxxxxx
```

"""

import argparse
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import write_info, EPISODES_STATS_PATH

V20 = "v2.0"
V21 = "v2.1"

video_path = None
episode_chunk = 0



def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
):
    global video_path
    dataset = LeRobotDataset(repo_id, revision=V21, force_cache_sync=True)
    dataset.consolidate(run_compute_stats=True)
    dataset.meta.info["codebase_version"] = V20
    write_info(dataset.meta.info, dataset.root)
    dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")
    # delete old episodes_stats.jsonl file
    if (dataset.root / EPISODES_STATS_PATH).is_file:
        (dataset.root / EPISODES_STATS_PATH).unlink()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))