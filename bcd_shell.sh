
source /root/miniconda3/etc/profile.d/conda.sh

conda env list 


conda activate llm_test


source /usr/local/Ascend/ascend-toolkit/set_env.sh


cd /models/share/chenyupeng/chenyupeng/nanogpt_optimizer_and_where_to_find_them/modded-nanogpt_record4_muon_improvements

python bcd_search.py --config configs/search_adamw.yaml