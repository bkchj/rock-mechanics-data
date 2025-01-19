clear
close  all
%% 数据读取
geshu=500%训练集的个数
%读取数据
shuru=xlsread('数据的输入.csv');
shuchu=xlsread('数据的输出.csv');
%nn = randperm(size(shuru,1));%随机排序
nn=1:size(shuru,1);%正常排序
input_train =shuru(nn(1:geshu),:);
input_train=input_train';
output_train=shuchu(nn(1:geshu),:);
output_train=output_train';
input_test =shuru(nn((geshu+1):end),:);
input_test=input_test';
output_test=shuchu(nn((geshu+1):end),:);
output_test=output_test';
%样本输入输出数据归一化
[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
global inputn outputn shuru_num shuchu_num
[inputn,inputps]=mapminmax('apply',input_train,bb);
[outputn,outputps]=mapminmax('apply',output_train,dd);
shuru_num = size(input_train,1); % 输入维度
shuchu_num = 1;  % 输出维度
%% 利用算法选择最佳的BP参数 
N = 10; %　巢穴数量(解决方案的规模)
D = 2; % 解的维数
T = 50 ; % 迭代次数
% 自变量上下限
ParticleScope(1,:)=[10 200];
ParticleScope(2,:)=[0.01 0.15];
ParticleScope=ParticleScope';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xv=rand(N,D); %首先，初始化种群个体速度和位置
for d=1:D
    xv(:,d)=xv(:,d)*(ParticleScope(2,d)-ParticleScope(1,d))+ParticleScope(1,d);  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nest_Pop=xv(:,1:D);  % 随机初始解决方案
Pa = 0.15 ; % 建新巢的概率(宿主鸟发现外来鸟蛋后)
for t=1:T
    levNestPop =  func_levy(Nest_Pop,ParticleScope) ;
    % 产生新的解    
    Nest_Pop = func_bestNestPop(Nest_Pop,levNestPop);  
    % 在新巢和旧巢中选择一个最好的巢 
    rand_nestPop = func_newBuildNest(Nest_Pop,Pa,ParticleScope);
    % 通过偏好随机游走放弃更糟糕的巢并建立新的巢
    Nest_Pop = func_bestNestPop(Nest_Pop,rand_nestPop) ; 
    % 在新巢和旧巢中选择一个最好的巢
    [~,index] = max(func_fitness(Nest_Pop)) ;
    % 最好的巢穴   
    trace(t) = func_objValue(Nest_Pop(index,:)) ; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 参数选择结果赋值
x=Nest_Pop(index,:);
zhongjian1_num = round(x(1));  
xue = x(2);
%% 模型建立与训练
layers = [ ...
    sequenceInputLayer(shuru_num)    % 输入层
    lstmLayer(zhongjian1_num)        % LSTM层
    fullyConnectedLayer(shuchu_num)  % 全连接层
    regressionLayer];
 
options = trainingOptions('adam', ...   % 梯度下降
    'MaxEpochs',80, ...                % 最大迭代次数
    'GradientThreshold',1, ...         % 梯度阈值 
    'InitialLearnRate',xue,...
    'Verbose',0, ...
    'Plots','training-progress');            % 学习率
%% 训练LSTM
net = trainNetwork(inputn,outputn,layers,options);
%% 预测
net = resetState(net);% 网络的更新状态可能对分类产生了负面影响。重置网络状态并再次预测序列。
[~,Ytrain]= predictAndUpdateState(net,inputn);
test_simu=mapminmax('reverse',Ytrain,dd);%反归一化
rmse = sqrt(mean((test_simu-output_train).^2));   % 训练集
%测试集样本输入输出数据归一化
inputn_test=mapminmax('apply',input_test,bb);
[net,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%反归一化
error1=test_simu1-output_test;%测试集预测-真实
%计算均方根误差 (RMSE)。
rmse1 = sqrt(mean((test_simu1-output_test).^2));  % 测试集
%% 画图

%将预测值与测试数据进行比较。
figure
plot(output_train)
hold on
plot(test_simu,'.-')
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("训练集")


figure
plot(output_test)
hold on
plot(test_simu1,'.-')
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("测试集")


 % 真实数据，行数代表特征数，列数代表样本数output_test = output_test;
T_sim_optimized = test_simu1;  % 仿真数据

num=size(output_test,2);%统计样本总数
error=T_sim_optimized-output_test;  %计算误差
mae=sum(abs(error))/num; %计算平均绝对误差
me=sum((error))/num; %计算平均绝对误差
mse=sum(error.*error)/num;  %计算均方误差
rmse=sqrt(mse);     %计算均方误差根
% R2=r*r;
tn_sim = T_sim_optimized';
tn_test =output_test';
N = size(tn_test,1);
R2=(N*sum(tn_sim.*tn_test)-sum(tn_sim)*sum(tn_test))^2/((N*sum((tn_sim).^2)-(sum(tn_sim))^2)*(N*sum((tn_test).^2)-(sum(tn_test))^2)); 

disp(' ')
disp('----------------------------------------------------------')

disp(['平均绝对误差mae为：            ',num2str(mae)])
disp(['平均误差me为：            ',num2str(me)])
disp(['均方误差根rmse为：             ',num2str(rmse)])
disp(['相关系数R2为：                ' ,num2str(R2)])




















