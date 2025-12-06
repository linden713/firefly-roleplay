import os
import json
import glob
import pandas as pd

def extract_metrics(report_path):
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    metrics = {
        'ooc_rate': data.get('ooc_rate'),
        'safety_violation_rate': data.get('safety_violation_rate'),
        'mean_ppl': data.get('mean_ppl'),
        'distinct_1': data.get('distinct', {}).get('distinct_1'),
        'distinct_2': data.get('distinct', {}).get('distinct_2'),
        'distinct_3': data.get('distinct', {}).get('distinct_3'),
        'repeat_rate_6gram': data.get('distinct', {}).get('repeat_rate_6gram'),
        'style_short_ratio_mean': data.get('style_short_ratio_mean'),
        'memory_hit_rate_ge1': data.get('memory_hit_rate_ge1'),
        'memory_hit_rate_ge2': data.get('memory_hit_rate_ge2'),
        'burstiness_mean': data.get('burstiness_mean'),
        'burstiness_var': data.get('burstiness_var'),
        'knowledge_consistency': data.get('knowledge_consistency')
    }
    return metrics

def main():
    base_dir = '/home/lch/firefly-roleplay/evaluation/evaluation_result'
    results = []

    for dirname in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dirname)
        if os.path.isdir(dir_path):
            # Find the latest report file
            report_files = glob.glob(os.path.join(dir_path, 'report_*.json'))
            if report_files:
                latest_report = max(report_files, key=os.path.getctime)
                metrics = extract_metrics(latest_report)
                metrics['experiment'] = dirname
                results.append(metrics)

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    
    # Reorder columns to put 'experiment' first
    cols = ['experiment'] + [c for c in df.columns if c != 'experiment']
    df = df[cols]
    
    # Sort by experiment name
    df = df.sort_values('experiment')

    # Print as Markdown table
    print(df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    main()
