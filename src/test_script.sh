#!/bin/bash

# get the all exp-id
data_dir="vanilla_ppo_ant"
config_file="config_ant_vanilla_ppo.json"
sqlite_file="ant_vanilla.db"
eps=0.15
model_name="sarsa_vanilla_ant.model"

cd ${data_dir}

exp_id=()
index=0
for file in $(ls *)
do
  exp_id[index]=$file
  index=$(($index+1))
done
cd -

for ((i=0;i<${#exp_id[@]};i++))
do
  echo "No attack"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --deterministic --sqlite-path $sqlite_file
done

echo "-------------------------------------------------------------------------------------"

for ((i=0;i<${#exp_id[@]};i++))
do
  echo "Critic attack"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method critic --deterministic --sqlite-path $sqlite_file
done

echo "-------------------------------------------------------------------------------------"

for ((i=0;i<${#exp_id[@]};i++))
do
  echo "Random attack"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method random --deterministic --sqlite-path $sqlite_file
done

echo "-------------------------------------------------------------------------------------"

for ((i=0;i<${#exp_id[@]};i++))
do
  echo "action attack"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method action --deterministic --sqlite-path $sqlite_file
done
echo "-------------------------------------------------------------------------------------"


for ((i=0;i<${#exp_id[@]};i++))
do
  echo "Sarsa Model Generate"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --sarsa-enable --sarsa-model-path ${model_name}
  echo "Done"
  echo "Begin to Test Sarsa attack"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method sarsa --attack-sarsa-network ${model_name} --deterministic --sqlite-path $sqlite_file
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method sarsa+action --attack-sarsa-network ${model_name} --deterministic --sqlite-path $sqlite_file
done
echo "-------------------------------------------------------------------------------------"