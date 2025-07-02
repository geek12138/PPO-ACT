for i in 41 # 40 41 42 43 44 45 46 47 48 49 50
do
python main_PPO.py -epochs 10000 -runs 1 \
    -L_num 200 -alpha 1e-3 -gamma 0.99 \
    -clip_epsilon 0.2 -question 2 -ppo_epochs 1 \
    -batch_size 1 -gae_lambda 0.95 \
    -delta 0.5 -rho 0.01 -seed ${i}
done
