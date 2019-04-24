"""Generate mean, median, standard deviations from bss eval metrics generated by museval.

usage: collect_eval_stats.py [options] <bss-eval-dir>

options:
    -h, --help                  Show this help message and exit
"""

from docopt import docopt
import glob
import json
import os

def compute_metrics(eval_path):
    files = glob.glob(os.path.join(eval_path, "*.json"))
    inst_list = None
    print("Found " + str(len(files)) + " JSON files to evaluate...")
    for path in files:
        if path.__contains__("test.json"):
            print("Found test JSON, skipping...")
            continue

        with open(path, "r") as f:
            js = json.load(f)

        if inst_list is None:
            inst_list = [list() for _ in range(len(js["targets"]))]

        for i in range(len(js["targets"])):
            inst_list[i].extend([np.float(f['metrics'][metric]) for f in js["targets"][i]["frames"]])

    inst_list = [np.array(perf) for perf in inst_list]

    if compute_averages:
        return [(np.nanmedian(perf), np.nanmedian(np.abs(perf - np.nanmedian(perf))), np.nanmean(perf), np.nanstd(perf)) for perf in inst_list]
    else:
        return inst_list


if __name__=="__main__":
    args = docopt(__doc__)
    bss_eval_dir = args["<bss-eval-dir>"]
    metrics = compute_metrics(os.path.join(bss_eval_dir,"*"))
    print(metrics)