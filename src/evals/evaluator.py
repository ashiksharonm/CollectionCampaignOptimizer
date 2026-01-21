import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict, Any
from src.bandits.policies import BasePolicy

class Evaluator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def compare_policies(self, results: List[Dict[str, Any]]):
        """
        Compare training results of multiple policies.
        """
        summary = []
        
        plt.figure(figsize=(12, 6))
        
        for res in results:
            name = res["model"]
            cum_reward = res["cumulative_reward"]
            avg_reward = np.mean(res["rewards"])
            total_reward = cum_reward[-1]
            
            summary.append({
                "model": name,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "final_regret": res["cumulative_regret"][-1]
            })
            
            plt.plot(cum_reward, label=f"{name} (Total: {total_reward:.2f})")
            
        plt.title("Cumulative Reward over Time")
        plt.xlabel("Rounds")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/cumulative_reward.png")
        plt.close()
        
        # Save summary
        # Calculate Uplift against Random if present
        random_base = next((x for x in summary if "Random" in x['model']), None)
        if random_base:
            base_reward = random_base['total_reward']
            for item in summary:
                item['uplift'] = (item['total_reward'] - base_reward) / abs(base_reward) * 100
        
        with open(f"{self.output_dir}/report.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Evaluation report saved to {self.output_dir}/report.json")
        return summary
    
    def plot_action_distribution(self, results: List[Dict[str, Any]]):
        """Plot action distribution for the final model"""
        for res in results:
            actions = res['actions']
            name = res['model']
            
            plt.figure(figsize=(8,4))
            sns.countplot(x=actions)
            plt.title(f"Action Distribution - {name}")
            plt.savefig(f"{self.output_dir}/actions_{name}.png")
            plt.close()
