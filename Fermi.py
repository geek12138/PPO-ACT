import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import asyncio

class SPGG(nn.Module):
    def __init__(self, L_num, device, r, epoches, 
                    now_time, question, output_path):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epoches = epoches
        self.question = question
        self.now_time = now_time
        self.output_path=output_path
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()


    def _init_state(self, question):
        if question == 1: # question 1: 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2: # question 2: 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:  # question 3: 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            # 创建一个 L x L 的零矩阵
            state = torch.zeros((self.L_num, self.L_num))
            # 填充交替的 0 和 1
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state(self, state_matrix):
        """将 2D 网格转换为 4D 张量后填充"""
        # 添加 batch 和 channel 维度 [B, C, H, W]
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 使用正确的填充参数格式 (padding_left, padding_right, padding_top, padding_bottom)
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')  # 四周各填充1
        
        # 计算邻域合作数
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L, L]
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)

    def calculate_reward(self, state_matrix):
        """
        state_matrix: 1表示C, 0表示D
        self.neibor_kernel: 中心为1，上下左右为1的5点卷积核
        """
        float_state = state_matrix.float().unsqueeze(0).unsqueeze(0)
        
        padded_state = F.pad(float_state, (1, 1, 1, 1), mode='circular')
        
        N_C_g = F.conv2d(padded_state, self.neibor_kernel)
        
        group_gross_profit = (self.r * N_C_g) / 5.0
        
        padded_gross = F.pad(group_gross_profit, (1, 1, 1, 1), mode='circular')
        total_gross_profit = F.conv2d(padded_gross, self.neibor_kernel).squeeze()
        
        total_cost = state_matrix.float() * 5.0

        reward_matrix = total_gross_profit - total_cost
        
        return reward_matrix

    def fermi_update(self, type_t_matrix):
        K = 0.1
        profit = self.calculate_reward(type_t_matrix)  # 现在profit已包含激励
        
        W_left = 1 / (1 + torch.exp((profit - torch.roll(profit, 1, 1))/K))
        W_right = 1 / (1 + torch.exp((profit - torch.roll(profit, -1, 1))/K))
        W_up = 1 / (1 + torch.exp((profit - torch.roll(profit, 1, 0))/K))
        W_down = 1 / (1 + torch.exp((profit - torch.roll(profit, -1, 0))/K))

        learning_direction = torch.randint(0,4,(self.L_num,self.L_num)).to(self.device)
        learning_probabilities = torch.rand(self.L_num,self.L_num).to(self.device)

        type_t1_matrix = (learning_direction==0)*((learning_probabilities<=W_left)*torch.roll(type_t_matrix,1,1)+(learning_probabilities>W_left)*type_t_matrix) +\
                        (learning_direction==1)*((learning_probabilities<=W_right)*torch.roll(type_t_matrix,-1,1)+(learning_probabilities>W_right)*type_t_matrix) +\
                        (learning_direction==2)*((learning_probabilities<=W_up)*torch.roll(type_t_matrix,1,0)+(learning_probabilities>W_up)*type_t_matrix) +\
                        (learning_direction==3)*((learning_probabilities<=W_down)*torch.roll(type_t_matrix,-1,0)+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(self.L_num,self.L_num)

    def run(self, num):
        coop_rates = []  # 合作率 = 当前状态中1的比例
        defect_rates = []  # 背叛率 = 当前状态中0的比例
        total_values = []  # 平均收益 = 所有个体的收益均值
        
        for epoch in tqdm(range(self.epoches)):
            self.epoch = epoch
            
            if epoch == 0:
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch, self.r, profit_matrix))
                # 计算当前指标（新增部分）
                current_coop_rate = self.current_state.float().mean().item()  # 合作率
                current_defect_rate = 1 - current_coop_rate  # 背叛率
                current_profit = self.calculate_reward(self.current_state).mean().item()  # 平均收益
                
                # 记录指标（新增部分）
                coop_rates.append(current_coop_rate)
                defect_rates.append(current_defect_rate)
                total_values.append(current_profit)
            
            # Fermi 更新   
            self.current_state = self.fermi_update(self.current_state)  

            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch+1, self.r, profit_matrix))

            # 计算当前指标（新增部分）
            current_coop_rate = self.current_state.float().mean().item()  # 合作率
            current_defect_rate = 1 - current_coop_rate  # 背叛率
            current_profit = self.calculate_reward(self.current_state).mean().item()  # 平均收益
            
            # 记录指标（新增部分）
            coop_rates.append(current_coop_rate)
            defect_rates.append(current_defect_rate)
            total_values.append(current_profit)

        return defect_rates, coop_rates, total_values

    def _save_snapshot(self, epoch, run_num):
        plt.figure(figsize=(8,8))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        # 设置 figure 的边框为黑色
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)  # 设置边框线宽
        plt.imshow(self.current_state.cpu().numpy(), cmap='gray_r')
        plt.title(f"Epoch {epoch}, Coop Rate: {self.current_state.float().mean().item():.2f}")
        # plt.savefig(f"{self.output_path}/snapshot_run{run_num}_epoch{epoch}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{self.output_path}/snapshot_run{run_num}_epoch{epoch}.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_data(self, data_type, name, r, run_num, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        # output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}_run{run_num}.txt', data)

    # def mkdir(self, path):
    #     os.makedirs(path, exist_ok=True)
    
    async def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保存策略矩阵快照与数据文件（与原Q-learning代码相同格式）"""
        plt.clf()
        plt.close("all")
        
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        # img_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        # matrix_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        # profit_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # =============================================
        # 1. 保存策略矩阵图
        # =============================================
        # 生成策略矩阵可视化
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        ax1.axis('off')
        # 设置 figure 的边框为黑色
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)  # 设置边框线宽
        
        # 创建颜色映射（黑:背叛者，白:合作者）
        color_map = {
            0: [0, 0, 0],    # 黑色
            1: [1, 1, 1]     # 白色
        }
        
        # 转换为RGB图像
        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color
        
        # 绘图设置
        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)
            
        # 保存图片
        # plt.savefig(f'{img_dir}/t={epoch}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        fig1.savefig(f'{img_dir}/t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)
        
        # =============================================
        # 2. 保存收益热图
        # =============================================
        # 2. 处理profit数据
        if isinstance(profit_data, tuple):
            # 如果是元组，提取combined_reward和team_utility
            combined_reward, _, team_utility = profit_data
            profit_matrix = combined_reward
        else:
            profit_matrix = profit_data
        
        # 确保是张量并转移到CPU
        if not isinstance(profit_matrix, torch.Tensor):
            profit_matrix = torch.tensor(profit_matrix, device=self.device)
        profit_matrix = profit_matrix.cpu().numpy()
        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)

        # 绘制热图
        vmin = 0
        vmax = np.ceil(np.maximum(5*(r-1),4*r))
        # profit_data = profit_matrix.cpu().numpy()
        im = ax2.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')
        
        # 添加颜色条
        
        cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=28)
        # cbar2.set_ticks(np.arange(vmin, vmax, 1))
        # cbar.set_label('Profit Value')
        cbar2.set_ticks(np.ceil(np.linspace(vmin, vmax, 5)).astype(int))
        
        # 设置坐标轴
        ax2.set_xticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.set_yticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.grid(False)

        ax2.axis('off')
        # 设置 figure 的边框为黑色
        # fig2.patch.set_edgecolor('black')
        # fig2.patch.set_linewidth(2)  # 设置边框线宽
        
        # 保存收益热图
        fig2.savefig(f'{img_dir}/profit_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

        # =============================================
        # 3. 保存矩阵数据文件
        # =============================================
        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                    type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                    profit_matrix, fmt='%.4f')
        return 0
