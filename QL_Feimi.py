import torch
from torch import tensor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import asyncio
import os
import re

class SPGG_Qlearning(nn.Module):
    def __init__(self,L_num,device,alpha,gamma,epsilon,r,
                    epoches,now_time,question,
                    is_QL, is_fermi, output_path,
                    lr=0.2,eta=0.8,count=0,cal_transfer=False):
        super(SPGG_Qlearning, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.alpha=alpha
        self.r=r
        self.gamma=gamma
        self.epsilon=epsilon
        self.cal_transfer=cal_transfer
        self.lr=lr
        self.eta=eta
        self.count=count
        self.neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
        self.now_time=now_time
        self.question=question

        self.is_QL=is_QL
        self.is_fermi=is_fermi
        self.output_path=output_path

        # Q表初始化（C略优）
        self.q_table = torch.randn((L_num, L_num, 2), device=device) * 0.5
        self.q_table += torch.tensor([[[2.0, 1.5]]], device=device)

    def indices_Matrix_to_Four_Matrix(self,indices):
        indices_left=torch.roll(indices,1,1)
        indices_right=torch.roll(indices,-1,1)
        indices_up=torch.roll(indices,1,0)
        indices_down=torch.roll(indices,-1,0)
        return indices_left,indices_right,indices_up,indices_down

    #update Q-table
    def updateQMatrix(self,alpha,gamma,type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):

        C_indices = torch.arange(type_t_matrix.numel()).to(self.device)

        A_indices = type_t_matrix.view(-1).long()

        B_indices = type_t1_matrix.view(-1).long()

        max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)

        update_values = (1 - self.eta) * Q_tensor[C_indices, A_indices, B_indices] + self.eta * (profit_matrix.view(-1) + gamma * max_values)

        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor

    def profit_Matrix_to_Four_Matrix(self,profit_matrix,K):
        W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
        W_right=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,1))/K))
        W_up=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,0))/K))
        W_down=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,0))/K))
        return W_left,W_right,W_up,W_down

    def fermiUpdate(self,type_t_matrix,type_t1_matrix,Q_tensor,profit_matrix):

        W_left,W_right,W_up,W_down=self.profit_Matrix_to_Four_Matrix(profit_matrix,0.1)

        learning_direction=torch.randint(0,4,(self.L_num,self.L_num)).to(self.device)

        learning_probabilities=torch.rand(self.L_num,self.L_num).to(self.device)

        type_t1_matrix=(learning_direction==0)*((learning_probabilities<=W_left)*torch.roll(type_t_matrix,1,1)+(learning_probabilities>W_left)*type_t_matrix) +\
                          (learning_direction==1)*((learning_probabilities<=W_right)*torch.roll(type_t_matrix,-1,1)+(learning_probabilities>W_right)*type_t_matrix) +\
                            (learning_direction==2)*((learning_probabilities<=W_up)*torch.roll(type_t_matrix,1,0)+(learning_probabilities>W_up)*type_t_matrix) +\
                                (learning_direction==3)*((learning_probabilities<=W_down)*torch.roll(type_t_matrix,-1,0)+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(self.L_num,self.L_num)


    def pad_matrix(self,type_t_matrix):

        tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)

        tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)
        return tensor_matrix

    def calculation_value(self, r, type_t_matrix):
        with torch.no_grad():
            pad_tensor = self.pad_matrix(type_t_matrix)
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(pad_tensor)
            coorperation_matrix = c_matrix.view(1, 1, self.L_num+2, self.L_num+2).to(torch.float32)

            coorperation_num = torch.nn.functional.conv2d(
                coorperation_matrix, 
                self.neibor_kernel,
                bias=None, 
                stride=1, 
                padding=0).view(self.L_num, self.L_num).to(self.device)

            c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)
            d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
            c_5_profit_matrix = torch.nn.functional.conv2d(
                c_profit_matrix.view(1, 1, self.L_num+2, self.L_num+2), 
                self.neibor_kernel,
                bias=None, 
                stride=1, 
                padding=0).to(torch.float32).to(self.device)

            d_5_profit_matrix = torch.nn.functional.conv2d(
                d_profit_matrix.view(1, 1, self.L_num+2, self.L_num+2), 
                self.neibor_kernel,
                bias=None, 
                stride=1, 
                padding=0).to(self.device)

            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t_matrix)
            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(self.L_num, self.L_num).to(torch.float32)

    def type_matrix_change(self,epsilon,type_matrix: tensor, Q_matrix: tensor):
        indices = type_matrix.long().flatten()
        Q_probabilities = Q_matrix[torch.arange(len(indices)), indices]

        max_values, _ = torch.max(Q_probabilities, dim=1)

        max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=self.device),
                                    torch.tensor(0.0, device=self.device))

        rand_tensor = torch.rand(max_tensor.size()).to(self.device)
        masked_tensor = (max_tensor.float() - (1 - max_tensor.float()) * 1e9).to(self.device)

        sum_tensor = (masked_tensor + rand_tensor).to(self.device)

        indices = torch.argmax(sum_tensor, dim=1).to(self.device)

        random_type = torch.randint(0,2, (self.L_num, self.L_num)).to(self.device)

        mask = (torch.rand(self.L_num, self.L_num, device=self.device) >= epsilon).long().to(self.device)

        updated_values = mask.flatten().unsqueeze(1) * indices.unsqueeze(1) + \
                        (1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)

        updated_tensor = updated_values.view(self.L_num, self.L_num).to(self.device)
        return updated_tensor

    def type_matrix_to_three_matrix(self,type_matrix: tensor):
        d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(self.device)
        c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(self.device)
        return d_matrix, c_matrix


    def generated_default_type_matrix(self):
        """
        生成由 0 和 1 组成的随机矩阵，控制 1 的比例为 `prob_1`
        Args:
            shape: 矩阵形状，如 (3,3)
            prob_1: 1 的比例，默认为 0.5
        Returns:
            torch.Tensor: 由 0 和 1 组成的矩阵，数据类型为 `torch.long`
        """
        prob_1=0.5
        matrix = torch.bernoulli(torch.full((self.L_num, self.L_num), prob_1))
        return matrix
        # return torch.tensor([1 /2, 1 / 2])

    def generated_default_type_matrix2(self):
        tensor = torch.zeros(self.L_num, self.L_num)
        mid_row = self.L_num // 2
        tensor[mid_row:, :] = 1
        return tensor

    def generated_default_type_matrix3(self):
        tensor = torch.zeros(self.L_num, self.L_num)
        return tensor

    def generated_default_type_matrix4(self):
        """
        生成由 0 和 1 组成的随机矩阵，控制 1 的比例为 `prob_1`
        Args:
            shape: 矩阵形状，如 (3,3)
            prob_1: 1 的比例，默认为 0.5
        Returns:
            torch.Tensor: 由 0 和 1 组成的矩阵，数据类型为 `torch.long`
        """
        prob_1=0.5
        # 生成均匀分布的随机数
        rand_matrix = torch.rand((self.L_num, self.L_num))
        # 根据阈值生成 0 和 1
        matrix = (rand_matrix < prob_1)
        return matrix
    
    def generated_default_type_matrix5(self):
        """
        生成由 0 和 1 组成的随机矩阵，严格保证 1 的比例为 `prob_1`
        Args:
            shape: 矩阵形状，如 (3,3)
            prob_1: 1 的比例，默认为 0.5
        Returns:
            torch.Tensor: 由 0 和 1 组成的矩阵，数据类型为 `torch.long`
        """
        prob_1=0.5
        total_elements = torch.prod(torch.tensor((self.L_num, self.L_num))).item()
        num_ones = int(round(total_elements * prob_1))
        # 生成全 0 矩阵
        matrix = torch.zeros((self.L_num, self.L_num))
        # 生成随机索引并填充 1
        indices = torch.randperm(total_elements)[:num_ones]
        matrix.view(-1)[indices] = 1
        return matrix

    def c_mean_v(self,value_tensor):
        positive_values = value_tensor[value_tensor > 0.0]
        mean_of_positive = torch.mean(positive_values)
        return mean_of_positive.item() + 1


    def c_mean_v2(self,value_tensor):
        # 创建布尔张量，表示大于零的元素
        positive_num = (value_tensor > 0).to(self.device)
        negetive_num = (value_tensor < 0).to(self.device)
        # 计算大于零的元素的均值
        mean_of_positive_elements = (value_tensor.to(torch.float32).sum()) / ((positive_num + negetive_num).sum())
        return mean_of_positive_elements.to("cpu")

    async def shot_pic(self,type_t_matrix: tensor,i,r,profit_matrix,num):
        plt.clf()
        plt.close("all")
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type_{num}'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type_{num}/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type_{num}/profit_matrix'
        
        # img_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        # matrix_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        # profit_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)
        # 初始化图表和数据
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        ax.axis('off')
        # 设置 figure 的边框为黑色
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)  # 设置边框线宽
        # 指定图的大小
        #             plt.figure(figsize=(500, 500))  # 10x10的图
        #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
        #             plt.colorbar(ticks=[0, 1, 2], label='Color')
        # 显示图片
        # 定义颜色映射
        color_map = {
            #0设置为黑色
            0: (0, 0, 0),  # 黑色
            #1设置为白色
            1: (255, 255, 255),  # 白色
        }
        image = np.zeros((self.L_num, self.L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        plt.imshow(image,interpolation='None')
        plt.imshow(image,interpolation='None')
        # generated1_2_file=f'data/Origin_Fermi_Qlearning3_q{str(self.question)}_{self.now_time}/shot_pic/r={str(r)}/two_type_{str(num)}/generated1_2'
        # self.mkdir(generated1_2_file)
        plt.savefig(f'{img_dir}/t={i}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{img_dir}/t={i}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        # self.mkdir(f'{generated1_2_file}/type_t_matrix')
        np.savetxt(f'{matrix_dir}/r={str(r)}_epoches={str(self.epoches)}_L={str(self.L_num)}_T={str(i)}_第{str(self.count)}次实验数据.txt',type_t_matrix.cpu().numpy())
        # self.mkdir(f'{generated1_2_file}/profit_matrix')
        np.savetxt(f'{profit_dir}/r={str(r)}_epoches={str(self.epoches)}_L={str(self.L_num)}_T={str(i)}_第{str(self.count)}次实验数据.txt',profit_matrix.cpu().numpy())
        plt.clf()
        plt.close("all")
        return 0


    def shot_pic2(self,type_t_matrix: tensor,i,r,num):
        plt.clf()
        plt.close("all")
        # 初始化图表和数据
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 4)
        color_map = {
            0:(128, 128, 128),
            1:(255, 255, 255),
            2:(0, 0, 0),
            3:(31,119,180)
        }
        image = np.zeros((self.L_num, self.L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        plt.title('Qlearning: '+f"T:{i}")
        plt.imshow(image,interpolation='None')
        #字符串全改成用f处理的
        generated1_file=f'data/Learning_and_Propagation_q{str(self.question)}_{self.now_time}/shot_pic/r={str(r)}/four_type_{str(num)}/generated1'
        self.mkdir(generated1_file)
        plt.savefig(f'{generated1_file}/t={i}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{generated1_file}/t={i}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        type_t_matrix_file=f'data/Learning_and_Propagation_q{str(self.question)}_{self.now_time}/shot_pic/r={str(r)}/four_type_{str(num)}/generated1/type_t_matrix'
        self.mkdir(type_t_matrix_file)
        torch.save(type_t_matrix.int(), f'{type_t_matrix_file}/r={str(r)}_epoches={str(self.epoches)}_L={str(self.L_num)}_T={str(i)}_{str(self.count)}times.txt')
        plt.clf()
        plt.close("all")

    def shot_save_data(self,type_t_minus_matrix: tensor,type_t_matrix: tensor,
                        type_t1_matrix: tensor,i,r,profit_matrix,Q_matrix,num):

        C_indices = torch.arange(type_t_matrix.numel()).to(self.device)

        A_indices = type_t_minus_matrix.view(-1).long()

        B_indices = type_t_matrix.view(-1).long()
        Q_sa_matrix = Q_matrix[C_indices, A_indices, B_indices].view(self.L_num, self.L_num)
        type_t_matrix_path=f'data/Learning_and_Propagation_q{str(self.question)}_{self.now_time}+ \
                    /shot_pic/r={str(r)}/two_type_{str(num)}/generated1_3/type_t_matrix'
        
        self.mkdir(type_t_matrix_path)
        np.savetxt(f'{type_t_matrix_path}/r={str(r)}_epoches={str(self.epoches)}_ \
                    L={str(self.L_num)}_T={str(i)}_{str(self.count)}times.txt',type_t_matrix.cpu().numpy())
        profit_matrix_path=f'data/Learning_and_Propagation_q{str(self.question)}_{self.now_time}+ \
                    /shot_pic/r={str(r)}/two_type_{str(num)}/generated1_3/profit_matrix'
        self.mkdir(profit_matrix_path)
        np.savetxt(f'{profit_matrix_path}/r={str(r)}_epoches={str(self.epoches)}_\
                    L={str(self.L_num)}_T={str(i)}_{str(self.count)}times.txt',profit_matrix.cpu().numpy())
        
        Q_sa_matrix_path=f'data/Learning_and_Propagation_q{str(self.question)}_{self.now_time}+ \
                            /shot_pic/r={str(r)}/two_type_{str(num)}/generated1_3/Q_sa_matrix'
        self.mkdir(Q_sa_matrix_path)
        np.savetxt(f'{Q_sa_matrix_path}/r={str(r)}_epoches={str(self.epoches)}_L={str(self.L_num)}_ \
                    T={str(i)}_{str(self.count)}times.txt',Q_sa_matrix.cpu().numpy())
        
        Q_matrix_path=f'data/Learning_and_Propagation_q{str(self.question)}_{self.now_time} \
                    /shot_pic/r={str(r)}/two_type_{str(num)}/generated1_3/Q_matrix'
        self.mkdir(Q_matrix_path)
        torch.save(Q_matrix,f'{Q_matrix_path}/r={str(r)}_epoches={str(self.epoches)}_\
                    L={str(self.L_num)}_T={str(i)}_{str(self.count)}times.txt')

    #计算CD比例和利润
    def cal_fra_and_value(self, D_Y, C_Y, D_Value, C_Value,all_value, type_t_minus_matrix,type_t_matrix, d_matrix, c_matrix, profit_matrix,i):
        # 初始化图表和数据

        d_value = d_matrix * profit_matrix
        c_value = c_matrix * profit_matrix
        dmean_of_positive = self.c_mean_v2(d_value)
        cmean_of_positive = self.c_mean_v2(c_value)
        count_0 = torch.sum(type_t_matrix == 0).item()
        count_1 = torch.sum(type_t_matrix == 1).item()
        D_Y = np.append(D_Y, count_0 / (self.L_num * self.L_num))
        C_Y = np.append(C_Y, count_1 / (self.L_num * self.L_num))
        D_Value = np.append(D_Value, dmean_of_positive)
        C_Value = np.append(C_Value, cmean_of_positive)
        all_value = np.append(all_value, profit_matrix.sum().item())
        CC, DD, CD, DC = self.cal_transfer_num(type_t_minus_matrix,type_t_matrix)
        return  D_Y, C_Y, D_Value, C_Value,all_value, count_0, count_1, CC, DD, CD, DC

    #计算转移的比例
    def cal_transfer_num(self,type_t_matrix,type_t1_matrix):
        CC=(torch.where((type_t_matrix==1)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (self.L_num * self.L_num)
        DD=(torch.where((type_t_matrix==0)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (self.L_num * self.L_num)
        CD=(torch.where((type_t_matrix==1)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (self.L_num * self.L_num)
        DC=(torch.where((type_t_matrix==0)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (self.L_num * self.L_num)
        return CC,DD,CD,DC

    #提取Qtable
    def extract_Qtable(self,Q_tensor, type_t_matrix):
        C_indices = torch.where(type_t_matrix.squeeze() == 1)[0]
        D_indices = torch.where(type_t_matrix.squeeze() == 0)[0]
        C_Q_table = Q_tensor[C_indices]
        D_indices = Q_tensor[D_indices]
        C_q_mean_matrix = torch.mean(C_Q_table, dim=0)
        D_q_mean_matrix = torch.mean(D_indices, dim=0)
        return D_q_mean_matrix.cpu().numpy(), C_q_mean_matrix.cpu().numpy()

    def split_four_policy_type(self,Q_matrix):
        CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0))
        DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                    Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
        CDC = torch.where((Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
        StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0))
        return DD.view((self.L_num,self.L_num)),CC.view((self.L_num,self.L_num)), CDC.view((self.L_num,self.L_num)), StickStrategy.view((self.L_num,self.L_num))

    def split_five_policy_type(self,Q_matrix,type_t_matrix):
        CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0)).view((self.L_num,self.L_num))
        DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                    Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0)).view((self.L_num,self.L_num))
        CDC = torch.where((Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0)).view((self.L_num,self.L_num))
        StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0)).view((self.L_num,self.L_num))
        CDC_C=CDC*torch.where(type_t_matrix==1,torch.tensor(1),torch.tensor(0))
        CDC_D=CDC*torch.where(type_t_matrix==0,torch.tensor(1),torch.tensor(0))
        CDC_neibor_num=0
        other_neibor_num=0
        CDC_neibor_DD, CDC_neibor_CC=torch.zeros((self.L_num,self.L_num)).to(self.device),torch.zeros((self.L_num,self.L_num)).to(self.device)
        if CDC.sum().item()!=0:
            CDC_neibor_matrix=self.pad_matrix(CDC.to(torch.float32).to(self.device))
            CDC_neibor_conv2d = torch.nn.functional.conv2d(CDC_neibor_matrix.view(1,1,self.L_num+2,self.L_num+2), self.neibor_kernel,
                                                            bias=None, stride=1, padding=0).view(self.L_num,self.L_num).to(self.device)
            CDC_neibor_num=(CDC_neibor_conv2d*CDC).sum().item()/CDC.sum().item()
            other_neibor_num = (CDC_neibor_conv2d * (1-CDC)).sum().item() / (1-CDC).sum().item()
            CDC_neibor_DD=torch.where(CDC_neibor_conv2d*(1-CDC)>0,torch.tensor(1),torch.tensor(0))*DD
            CDC_neibor_CC=torch.where(CDC_neibor_conv2d*(1-CDC)>0,torch.tensor(1),torch.tensor(0))*CC
        return DD,CC, CDC, StickStrategy,CDC_D,CDC_C,CDC_neibor_num,other_neibor_num,CDC_neibor_DD,CDC_neibor_CC

    def cal_four_type_value(self,DD,CC,CDC,StickStrategy,profit_matrix):
        CC_value = profit_matrix * CC
        DD_value = profit_matrix * DD
        CDC_value = profit_matrix * CDC
        StickStrategy_value = profit_matrix * StickStrategy
        return  DD_value,CC_value, CDC_value, StickStrategy_value

    def cal_five_type_value(self,DD,CC,CDC,StickStrategy,CDC_D,CDC_C,CDC_neibor_DD,CDC_neibor_CC,profit_matrix):
        CC_value = profit_matrix * CC
        DD_value = profit_matrix * DD
        CDC_value = profit_matrix * CDC
        StickStrategy_value = profit_matrix * StickStrategy
        CDC_C_value = profit_matrix * CDC_C
        CDC_D_value = profit_matrix * CDC_D
        CDC_neibor_DD_value = profit_matrix * CDC_neibor_DD
        CDC_neibor_CC_value = profit_matrix * CDC_neibor_CC
        return  DD_value,CC_value, CDC_value, StickStrategy_value,CDC_D_value,CDC_C_value,CDC_neibor_DD_value,CDC_neibor_CC_value

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        output_dir = f'{self.output_path}/{type}'
        os.makedirs(output_dir, exist_ok=True)
        # output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}_run{count}.txt', data)

    def extra_Q_table(self,loop_num):
        for i in range(loop_num):
            Q_matrix,type_t_matrix = self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
            D_q_mean_matrix, C_q_mean_matrix = self.extract_Qtable(Q_matrix, type_t_matrix)
            print(D_q_mean_matrix,C_q_mean_matrix)
            self.save_data('D_Qtable', 'D_Qtable',self.r, str(i), D_q_mean_matrix)
            self.save_data('C_Qtable', 'C_Qtable',self.r, str(i), C_q_mean_matrix)

    # # 早期激励，在前1000步增加合作奖励
    # def early_stage_bonus(self, payoff,epoch):
    #     if epoch < 1000:
    #         payoff[:,0] += 2.0  # 给合作者额外奖励
    #     return payoff
    
    # # 随机突变，在每100步执行随机突变
    # def mutation(self,type_t_matrix,epoch):
    #     if epoch % 100 == 0:
    #         mutate_mask = torch.rand_like(type_t_matrix) < 0.001
    #         type_t_matrix[mutate_mask] = 1 - type_t_matrix[mutate_mask]  # 翻转策略
    #         self.q_table[mutate_mask] = torch.rand(2, device=self.device) * 3  # 重置Q表
    def run(self,num):
        # 初始化类型矩阵和Q表
        if self.question==1:
            type_t_matrix = self.generated_default_type_matrix().to(self.device)
        elif self.question==2:
            type_t_matrix = self.generated_default_type_matrix2().to(self.device)
        else:
            type_t_matrix = self.generated_default_type_matrix3().to(self.device)

        Q_matrix = torch.zeros((self.L_num * self.L_num, 2, 2), dtype=torch.float32).to(self.device)
        
        # 数据记录容器
        D_Y, C_Y = np.array([]), np.array([])
        D_Value, C_Value = np.array([]), np.array([])
        all_value = np.array([])

        for i in tqdm(range(self.epoches)):
            current_epsilon = self.epsilon * (0.95 ** (i // 1000))
            rand_mask = torch.rand_like(type_t_matrix) < current_epsilon
            type_t_matrix[rand_mask] = torch.randint(0, 2, size=type_t_matrix[rand_mask].shape, device=self.device, dtype=torch.float32)
            # 计算利润矩阵
            profit_matrix = self.calculation_value(self.r, type_t_matrix)
            # 收益计算（带早期激励）
            # payoff = self.calculate_payoff()
            # profit_matrix = self.early_stage_bonus(profit_matrix,i)
            # 突变机制
            # self.mutation(type_t_matrix,i)

            # 保存关键步的快照
            if i+1 in [1, 10, 100, 1000, 10000, 100000]:
                asyncio.create_task(self.shot_pic(type_t_matrix, i+1, self.r, profit_matrix, num))
            
            # Q-learning策略更新
            type_t1_matrix = self.type_matrix_change(self.epsilon, type_t_matrix, Q_matrix)
            
            # Fermi邻居传播更新
            if self.is_fermi:
                type_t1_matrix = self.fermiUpdate(type_t_matrix, type_t1_matrix, Q_matrix, profit_matrix)

            # 更新Q表
            Q_matrix = self.updateQMatrix(self.alpha, self.gamma, type_t_matrix, type_t1_matrix, 
                                        Q_matrix, profit_matrix)
            
            # 记录数据
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t_matrix)
            D_Y, C_Y, D_Value, C_Value, all_value, *_ = self.cal_fra_and_value(
                D_Y, C_Y, D_Value, C_Value, all_value, type_t_matrix, type_t1_matrix,
                d_matrix, c_matrix, profit_matrix, i)
            
            # 更新类型矩阵
            type_t_matrix = type_t1_matrix.clone()

        return D_Y, C_Y, D_Value, C_Value, all_value