import argparse

from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_csv', type=str, default='./eval.csv')
    args = parser.parse_args()

    max_log_len = 0
    max_metric_len = 0
    avg = defaultdict(lambda: defaultdict(list))
    with open(args.log_csv, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            log_name, metric_name, value = line.split(',')
            log_name = log_name.strip()[:-2]
            metric_name = metric_name.strip()
            value = float(value.strip())
            avg[log_name][metric_name].append(value)
            max_log_len = max(max_log_len, len(log_name))
            max_metric_len = max(max_metric_len, len(metric_name))
    print(
        f'{"log_name":{max_log_len}s} {"metric_name":{max_metric_len}s} value')
    for log_name, metrics in avg.items():
        for metric_name, values in metrics.items():
            avg[log_name][metric_name] = sum(values) / len(values)
            print(f'{log_name:{max_log_len}s} {metric_name:{max_metric_len}s}'
                  f' {avg[log_name][metric_name]}')
