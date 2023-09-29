from scipy import stats
import pickle
import numpy as np
import sqlite3

# methods = ["care", "mhead", "pcgrad", "pure", "actor"]
methods = ["Vanilla", "SA-PPO(SGLD)", "SA-PPO(Convex)", "Radial", "Ours"]
names = ["vanilla", "sgld", "convex", "radial", "rluc"]
data = []
envs = ["hopper", "walker", "humanoid", "halfcheetah"]

for env in envs:
    print("-" * 80)
    print(env)
    data = []
    for idx in range(len(methods)):
        print("*" * 80)
        path = "./result_db/" + env + "/" + env + "_" + names[idx] + "_0-15.db"
        print(methods[idx])
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        sql = """SELECT mean_reward FROM attack_results WHERE method='sarsa+action'"""
        cursor.execute(sql)
        results = cursor.fetchall()
        print([r[0] for r in results])
        data.append([r[0] for r in results])
        conn.close()
    print("*"*80)
    n = len(data)
    for i in range(len(data)-1):
        print(stats.ttest_ind(data[i], data[n-1]))








