for i in 1
do
python main_PPO_ACT.py -epochs 10000 -runs 1 \
    -L_num 200 -alpha 1e-2 -gamma 0.96 \
    -clip_epsilon 0.2 -question 1 -ppo_epochs 1 \
    -batch_size 1 -gae_lambda 0.95 \
    -delta 0.5 -rho 0.001 -seed 41 \
    -pretrained_path data/PPO_2025_07_02_182448_q2_e_10000_L_200_a_0.001_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.01_seed_41/checkpoint/model_r5.0_epoch1000.pth
done
