import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class StatisticsVisualizer:
    def __init__(self):
        self.image_stats = defaultdict(lambda: defaultdict(int))
        self.total_stats = defaultdict(int)

    def add_image_results(self, image_name, detections):
        for det in detections:
            if det.get('classification_success', False):
                label = det.get('label', 'unknown')
                self.image_stats[image_name][label] += 1
                self.total_stats[label] += 1

    def generate_csv(self, output_path):
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['图片名', '标志类别', '数量'])
            
            all_labels = set(self.total_stats.keys())
            for image_name in sorted(self.image_stats.keys()):
                img_stats = self.image_stats[image_name]
                for label in sorted(img_stats.keys()):
                    count = img_stats[label]
                    if count > 0:
                        writer.writerow([image_name, label, count])
            
            writer.writerow([])
            writer.writerow(['统计汇总', '', ''])
            for label in sorted(self.total_stats.keys()):
                writer.writerow(['总计', label, self.total_stats[label]])
        
        print(f"CSV文件已保存: {output_path}")

    def generate_histogram(self, output_path, figsize=(10, 6)):
        if not self.total_stats:
            print("没有统计数据可用于生成直方图")
            return
        
        labels = sorted(self.total_stats.keys())
        counts = [self.total_stats[label] for label in labels]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(labels, counts, color=['red', 'blue', 'yellow', 'gray', 'green'][:len(labels)], edgecolor='black')
        
        ax.set_xlabel('标志类别', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title('交通标志统计直方图', fontsize=14, fontweight='bold')
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, fontsize=10)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"直方图已保存: {output_path}")

    def print_summary(self):
        print("\n" + "="*50)
        print("检测结果统计汇总")
        print("="*50)
        
        for image_name in sorted(self.image_stats.keys()):
            print(f"\n图片: {image_name}")
            img_stats = self.image_stats[image_name]
            for label in sorted(img_stats.keys()):
                count = img_stats[label]
                if count > 0:
                    print(f"  - {label}: {count} 个")
        
        print("\n" + "-"*50)
        print("总计统计:")
        for label in sorted(self.total_stats.keys()):
            print(f"  - {label}: {self.total_stats[label]} 个")
        
        total_all = sum(self.total_stats.values())
        print(f"\n总共检测到 {total_all} 个标志牌")
        print("="*50 + "\n")
