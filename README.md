是在 https://github.com/brightmart/albert_zh 基础上改的

基础模型参考的是albert_tiny_zh_google


目的：微调albert作为encoder，获得文本的语义表示


模型结构图参考论文Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks，具体如下：

<img src="https://github.com/34127chi/albert_zh_sr/blob/master/image/模型结构.png"  width="90%" height="70%" />

包括模型的训练、验证、导出，示例分别如下：
python run_classifier_sp_google_sr.py --task_name=lcqmc_pair --do_train=true    --data_dir=./lcqmc   --vocab_file=./albert_tiny_zh_google/vocab.txt   --albert_config_file=./albert_tiny_zh_google/albert_config_tiny_g.json  --train_batch_size=64   --num_train_epochs=40     --output_dir=./albert_lcqmc_tiny_google_checkpoints --init_checkpoint=./albert_lcqmc_tiny_google_checkpoints/

python run_classifier_sp_google_sr.py --task_name=lcqmc_pair --do_eval=true    --data_dir=./lcqmc   --vocab_file=./albert_tiny_zh_google/vocab.txt   --albert_config_file=./albert_tiny_zh_google/albert_config_tiny_g.json --output_dir=./albert_lcqmc_tiny_google_checkpoints --init_checkpoint=./albert_lcqmc_tiny_google_checkpoints/


python run_classifier_sp_google_sr.py --task_name=lcqmc_pair --do_export=true    --data_dir=./lcqmc   --vocab_file=./albert_tiny_zh_google/vocab.txt   --albert_config_file=./albert_tiny_zh_google/albert_config_tiny_g.json --output_dir=./albert_lcqmc_tiny_google_checkpoints --init_checkpoint=./albert_lcqmc_tiny_google_checkpoints/ --export_dir=./model/


导出模型的加载测试代码:
python client.py

结果指标：皮尔逊系数是0.714
