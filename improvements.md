# MahjongGB改进

以下是该项目在基线基础上的改进

## 1. 特征工程

原始版本的观测空间大小 (`OBS_SIZE`) 为 6。改进版本显著扩展至 154。

**新增特征详述**：
1. **副露信息 (`PACKS`)**: 记录了每个玩家（包括自己）的吃、碰、明杠的牌面信息。特征维度从 `OFFSET_OBS['PACKS']` (6) 开始，每个玩家占据7个通道（channel），共4个玩家，涉及如吃、碰、杠的类型以及来源等信息。
2. **自家暗杠信息 (`MY_ANGANG`)**: 专门的特征通道 (`OFFSET_OBS['MY_ANGANG']` = 34) 记录自家暗杠的牌。
3. **对手暗杠数量 (`OPPONENT_ANGANG_COUNT`)**: 记录其他三个对手各自暗杠的数量（`OFFSET_OBS['OPPONENT_ANGANG_COUNT']` = 35, 占据3个通道），但不揭示具体牌面。
4. **历史出牌记录 (`HISTORY`)**: 引入了对所有玩家历史出牌的记录 (从 `OFFSET_OBS['HISTORY']` = 38 到 `OFFSET_OBS['REMAINING']` = 150 之前)。每个玩家的出牌历史被编码为一个固定长度的序列（每个玩家28步历史，每步历史用一个通道表示牌的ID），这使得模型可以捕捉到出牌的时序信息和对手的出牌习惯。
5. **剩余牌墙信息 (`REMAINING`)**: 根据当前手牌、副露、打出的牌等信息，估算牌墙中每种牌剩余的数量（具体到剩余1张、2张、3张、4张，从 `OFFSET_OBS['REMAINING']` = 150 开始，占据4个通道）。
6. **特征处理函数更新**：相应地，增加了 `_packs_embedding_update`、`_history_embedding_update`、`_update_true_remaining_tiles_feature` 等函数来生成和更新这些复杂的特征嵌入。`_hand_embedding_update` 也随之调整以适应新的特征布局。

## 2. 模型架构

为了有效处理更丰富的特征，特别是时序相关的历史出牌记录，模型架构也从简单的CNN演进为更复杂的混合网络结构：

**模型架构**：

将154维的输入特征划分为两大部分：
*   **静态特征**: 包括手牌、副露、风牌、自家暗杠、对手暗杠数、剩余牌墙等。具体来说是 `obs[:, :38]` 和 `obs[:, 150:]` 连接而成，共 `38 + (154-150) = 42` 个通道。这42个通道的静态特征通过一个专门的CNN (`static_cnn`) 进行处理。该CNN包含普通卷积层、ResBlock和BatchNorm。CNN部分最终输出一个固定维度的静态特征嵌入 (`static_embedding`)。
*   **时序特征 (历史记录)**: 即 `obs[:, 38:150]`，这部分是各玩家的历史出牌记录。针对历史出牌记录 (每个玩家28步，共4个玩家，即112个历史步骤，`history_input = obs[:, 38:150]`，取其 `[:, :, 0, 0]` 作为牌的ID序列，形状为 `[B, 112]`)，引入了 Transformer Encoder (`history_transformer`)。在112步历史序列前加入一个可学习的分类标记 `[CLS]` (ID为`num_tiles + 1`)，序列长度变为113。将历史打出的每一张牌的ID（包括填充符0、CLS token和34种牌）通过一个Embedding层 (`history_tile_embedding`) 映射为高维向量 (`d_model`维)。除此之外，还为Transformer的输入序列添加位置信

从CNN部分获取的静态特征嵌入 (`static_embedding`) 与从Transformer部分获取的历史特征嵌入 (`history_embedding`) 进行拼接，形成一个融合了全局信息和时序信息的综合特征表示 (`fused_embedding`)。融合后的特征再分别通过不同的全连接层（带有BatchNorm1d和Dropout）输出策略（`_logits`）和价值（`_value_branch`）。改进了网络权重初始化方法 (`_initialize_weights`)，对卷积层、线性层、BatchNorm层以及Transformer的特定权重进行了细致的初始化。并更广泛地使用了 `BatchNorm1d/2d` 和 `Dropout/Dropout2d` 来提高训练稳定性和模型的泛化能力。

## 3. 奖励函数设计

在原始版本仅基于最终得分的稀疏奖励基础上，设计了更为精细和密集的中间奖励：

**中间奖励机制**：
*   **非线性向听数奖励/惩罚 (`SHANTEN_DECREASE_REWARD_BASE`, `SHANTEN_INCREASE_PENALTY_BASE`, `SHANTEN_NON_LINEAR_FACTOR`)**:当智能体的动作（摸牌、吃、碰、杠、打牌后）使其手牌向听数减少时，给予正奖励。奖励的大小与向听数减少的量成正比，并且引入了非线性因子 (`SHANTEN_NON_LINEAR_FACTOR / (after_shanten + 1)`)，使得越接近听牌（`after_shanten`越小），降低向听数的奖励值越高。反之，如果动作导致向听数增加，则给予负奖励（惩罚）。
*   **牌型潜力奖励 (`FAN_POTENTIAL_REWARD`)**:不仅仅关注整体向听数（通过 `MahjongShanten` 计算），还关注特定大番数牌型（如“对对和”用`RegularShanten`、“七对子”用`SevenPairsShanten`、“清一色”/“混一色”用自定义的`_calculate_flush_shanten`）的向听数。当智能体的动作使得这些高价值牌型的向听数减少时，给予额外的正奖励。这鼓励智能体在有机会时尝试做大牌。通过 `_get_hand_details` 函数获取当前手牌的各种向听数信息。
*   **精炼副露惩罚与奖励 (`MELD_PENALTY_BASE`, `MELD_SHANTEN_BONUS`)**:对吃 (`_chow`)、碰 (`_pung`) 等副露行为施加一个小的基础惩罚 (`MELD_PENALTY_BASE`)，因为副露会减少手牌的灵活性并暴露信息。但是，如果该副露行为成功地降低了手牌的向听数（比较副露前后的 `main_shanten`），则给予一个奖励 (`MELD_SHANTEN_BONUS` * 向听数减少量)，该奖励可以抵消甚至超过基础惩罚，从而鼓励有效的副露。
*   **防守奖励与惩罚 (`SAFE_TILE_REWARD`, `DANGEROUS_TILE_PENALTY`, `IMMEDIATE_RON_PENALTY_FACTOR`)**:打出安全牌（其他玩家已经打出过的牌，即“现物”）时，在 `_discard` 方法中给予少量正奖励 (`SAFE_TILE_REWARD`)。通过 `_is_tile_safe` 判断打出的牌是否为现物（检查 `self.discards` 中是否存在）。打出非安全牌（“生张”）时，给予少量负奖励 (`DANGEROUS_TILE_PENALTY`)。如果打出的牌直接导致其他玩家荣和（即“放铳”），则根据荣和的番数施加一个显著的负奖励（在 `_checkMahjong` 中，惩罚大小为 `fanCnt * IMMEDIATE_RON_PENALTY_FACTOR`）。
*   **有效牌奖励 (`EFFECTIVE_TILES_REWARD`)**:根据手牌有效牌（`RegularShanten` 返回的 `u`）数量的变化给予奖励/惩罚。

这些中间奖励，使得智能体在学习过程中能获得更频繁、更具体的反馈，从而加速学习进程，并引导其学习到更高级的攻防策略。

## 4. 学习过程与算法

Learner负责模型的更新。改进版本在标准PPO算法的基础上引入了预训练和模仿学习机制：

在Learner进程启动时，会尝试从指定路径 (`self.config.get('ckpt_save_path', 'model/')` + `pretrained_rl_cnn.pkl`) 加载一个预训练好的模型权重。这个预训练模型通常是通过监督学习（例如，使用 `pretrain_rl_model.py` 脚本在人类对局数据上训练）得到的。加载预训练权重可以为强化学习提供一个更好的初始策略，避免从完全随机的策略开始探索，从而加速收敛。预训练权重会同时加载到主模型 (`model`) 和专家模型 (`expert_model`)。

除了RL模型（即正在训练的策略网络和价值网络）外，Learner还维护一个参数固定的“专家模型”（`expert_model`）。这个专家模型的权重也从加载的预训练模型初始化，并且在RL训练过程中其参数**不会被更新** (`expert_model.eval()` 和 `param.requires_grad = False`)。

在计算PPO的损失函数时，额外增加了一项模仿学习损失。具体做法是：
1.  获取当前RL模型对于一个batch内状态的策略输出（logits）。
2.  获取专家模型对于相同状态的策略输出（logits）。
3.  计算RL模型策略输出的log-softmax (`rl_log_probs`) 和专家模型策略输出的softmax (`expert_probs`)。
4.  计算这两个分布之间的**KL散度 (Kullback-Leibler divergence)** (`F.kl_div(rl_log_probs, expert_probs, reduction='batchmean')`)，作为模仿学习损失。总的损失函数变为：`loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss + self.config['imitation_coeff'] * imitation_loss`。其中 `imitation_coeff` 是一个超参数，用于控制模仿学习损失项在总损失中的权重。

模仿学习损失项的引入，旨在鼓励RL智能体在通过与环境交互进行探索的同时，其策略不要过分偏离专家模型所代表的（可能更优的）行为模式。这有助于稳定训练过程，防止策略在探索过程中出现灾难性遗忘或陷入局部最优，并可能引导智能体学习到更接近专家水平的策略。

## 5. 数据预处理、增强与预训练 (新增模块)

**`improved/preprocess.py` (数据预处理)**：

该脚本负责将原始的、可能是文本格式的人类对局日志 (`data/data.txt`) 转换为结构化的训练数据。

**流程**：
1.  逐行读取原始数据，模拟牌局进程。
2.  使用 `FeatureAgent` 将每个决策点的牌局状态转换为模型所需的观测和动作掩码。
3.  记录玩家在该状态下实际采取的动作。
4.  记录该局的最终得分，并转换为一个简化的结果标签（outcome，赢家为1，输家为-1，其他人为0）。
5.  **数据过滤 (`filterData`)**：在保存前，会过滤掉那些只有一个合法动作的样本（即动作掩码中只有一个1）。这是因为模型无法从这种没有选择余地的状态中学习到策略。
6.  将处理好的 `obs`, `mask`, `act`, `outcome` 数据打包保存为 NumPy的 `.npz` 格式文件 (存入 `data/` 目录)，每个文件对应一个或多个原始对局。同时生成一个 `data/count.json` 文件记录每个 `.npz` 文件包含的样本数量。

**`improved/dataset.py` (数据加载与增强)**：
*   **`MahjongGBDataset` 类**负责从 `preprocess.py` 生成的 `.npz` 文件中加载数据 (可指定 `data_dir_prefix`)。支持按比例划分训练集和验证集 (通过 `begin` 和 `end` 参数)。
*   **`AugmentedMahjongGBDataset` 类**:包装 `MahjongGBDataset` 对象，并对其提供的样本进行**数据增强 (Data Augmentation)**。

数据增强方法包括：
*   `_swap_suits`: 随机交换万、筒、条三种花色在特征中的位置，并相应调整动作和掩码。
*   `_mirror_numbers`: 对数字牌1-9进行镜像反转（例如1变9，2变8等），并调整动作和掩码。
*   `_shuffle_hand`: 随机打乱手牌在内部特征表示中的顺序（不改变实际手牌组合，仅影响观测中手牌通道的顺序）。
*   `_rotate_winds`: 随机旋转东、南、西、北风（作为门风或场风时）在特征中的位置，并调整动作和掩码。

**`improved/pretrain_rl_model.py` (监督学习预训练)**：

该脚本使用 `dataset.py` 加载的专家数据，对 `improved/model.py` 中定义的CNN+Transformer混合模型进行监督学习预训练。

**训练流程**：
1.  加载训练 (`MahjongGBDataset` + `AugmentedMahjongGBDataset`) 和验证 (`MahjongGBDataset`) 数据集。
2.  初始化模型、优化器（Adam）和学习率调度器 (`WarmupCosineScheduler`，一种先预热再余弦衰减的学习率策略）。
3.  **损失函数**：同时优化策略和价值。
    *   **策略损失**：使用交叉熵损失函数 (`F.cross_entropy`)，目标是预测专家在该状态下采取的动作。
    *   **价值损失**：使用均方误差损失函数 (`F.mse_loss`)，目标是预测该状态最终的局点结果。
    *   总损失是这两者加权和（`value_loss_weight` 控制价值损失的权重）。
4.  标准的训练循环：前向传播、计算损失、反向传播、更新参数。
5.  包含验证集上的评估，用于监控模型性能和选择最佳模型。
6.  保存训练日志 (`training_log_{timestamp}.log`)、训练历史 (`training_history.json`)、每个epoch的检查点、最佳模型检查点 (`best_model.pkl`) 以及最终的预训练模型 (`pretrained_rl_cnn.pkl`) 到 `model/` 目录。

预训练得到的模型 (`pretrained_rl_cnn.pkl`) 将作为后续强化学习阶段 (在 `learner.py` 中) 的初始模型和模仿学习的目标专家模型。这使得RL的起点不再是随机策略，而是已经具备一定麻将知识的策略。