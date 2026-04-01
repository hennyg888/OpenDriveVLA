import json
import numpy as np
import collections
from operator import itemgetter
from drivevla.utils.trajectory_utils import retrieve_traj

def main():
    conv_data_list = []
    object_agent_trajectory_file = 'output/stage3_object_agent_trajectory_on_QA_20250304_013305/planning_conversations_val.json'
    
    # 过滤轨迹
    with open(object_agent_trajectory_file, 'r') as f:
        for line in f:
            conv_data = json.loads(line.strip())
            end_pos = retrieve_traj(conv_data['answer'][0])[-1]
            if np.sqrt(end_pos[0]**2 + end_pos[1]**2) > 0.2:
                conv_data_list.append(conv_data)
    
    with open(object_agent_trajectory_file.replace('.json', '_filtered.json'), 'w') as f:
        for conv_data in conv_data_list:
            f.write(json.dumps(conv_data) + '\n')
    
    print(f"过滤后的数据数量: {len(conv_data_list)}")
    
    # 统计每个sample token出现的次数
    sample_counts = count_sample_tokens("/home/feng/xuyuan/DriveVLA/drivevlms.worktrees/planning-oriented/llava-next/uniad/output/stage3_object_agent_trajectory_on_QA_20250304_013305/planning_conversations_val_filtered.json")
    
    # 打印结果
    print_sample_stats(sample_counts)

def count_sample_tokens(file_path):
    """统计文件中每个sample token出现的次数"""
    sample_counts = collections.defaultdict(int)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                conv_data = json.loads(line.strip())
                if 'id' in conv_data:
                    # 提取sample token (第一个_前的部分)
                    sample_token = conv_data['id'].split('_')[0]
                    sample_counts[sample_token] += 1
            except json.JSONDecodeError as e:
                print(f"解析错误: {e}")
                continue
    
    # 转换为列表并按计数排序（从高到低）
    sorted_counts = sorted(sample_counts.items(), key=itemgetter(1), reverse=True)
    
    return sorted_counts

def print_sample_stats(sorted_counts):
    """打印并保存统计结果"""
    # 计算总样本数和唯一样本数
    total_samples = sum(count for _, count in sorted_counts)
    unique_samples = len(sorted_counts)
    
    print(f"\n统计结果:")
    print(f"总共找到 {total_samples} 条数据")
    print(f"共有 {unique_samples} 个不同的sample token")
    
    print("\n出现次数最多的20个sample token:")
    for token, count in sorted_counts[:20]:
        print(f"Sample: {token}, 出现次数: {count}")
    
    # 统计不同计数的分布
    count_distribution = collections.defaultdict(int)
    for _, count in sorted_counts:
        count_distribution[count] += 1
    
    print("\n计数分布:")
    for count in sorted(count_distribution.keys()):
        print(f"出现 {count} 次的sample token数量: {count_distribution[count]}")
    
    # 将结果写入文件
    output_file = 'sample_token_counts.json'
    with open(output_file, 'w') as f:
        json.dump(sorted_counts, f, indent=2)
    
    print(f"\n完整结果已保存到 {output_file}")

if __name__ == '__main__':
    main()