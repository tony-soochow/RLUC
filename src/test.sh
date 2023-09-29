#!/bin/bash
 # get the all exp-id
data_dir=$1
config_file=$2
sqlite_file=$3
eps=$4
sarsa_model_name=$5
imit_model_name=$6

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
  echo "-------------------No attack----------------------------"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --deterministic --sqlite-path $sqlite_file
  echo "-------------------Critic attack----------------------------"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method critic --deterministic --sqlite-path $sqlite_file
  echo "-------------------Random attack----------------------------"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method random --deterministic --sqlite-path $sqlite_file
  echo "-------------------action attack----------------------------"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method action --deterministic --sqlite-path $sqlite_file

  echo "Sarsa Model Generate"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --sarsa-enable --sarsa-model-path ${sarsa_model_name}
  echo "Done"
  echo "-------------------sarsa attack----------------------------"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method sarsa --attack-sarsa-network ${sarsa_model_name} --deterministic --sqlite-path $sqlite_file
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method sarsa+action --attack-sarsa-network ${sarsa_model_name} --deterministic --sqlite-path $sqlite_file


  echo "imit Model Generate"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --imit-enable --imit-model-path ${imit_model_name}
  echo "Done"
  echo "-------------------imit+action attack----------------------------"
  python test.py --config-path $config_file --exp-id ${exp_id[$i]} --attack-eps=${eps} --attack-method action+imit --imit-model-path ${imit_model_name} --deterministic --sqlite-path $sqlite_file
done







