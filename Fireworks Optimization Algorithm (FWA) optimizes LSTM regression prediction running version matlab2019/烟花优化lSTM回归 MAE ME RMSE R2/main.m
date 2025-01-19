clear
close  all
%% 数据读取
geshu=900;%训练集的个数
%读取数据
shuru=xlsread('数据的输入.csv');
shuchu=xlsread('数据的输出.csv');
nn = randperm(size(shuru,1));%随机排序
% nn=1:size(shuru,1);%正常排序
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
%%
N=5;   % N烟花数
T=10;   % T为迭代次数
D=2;     % D变量维数


M=5;     % M变异火花数
En=6;    % En爆炸数目
Er=5;    % Er爆炸半径
a=0.3;   % a,b为爆炸数目限制因子
b=0.6;

%求最大值变量上下界
LB=[10 0.01 ];
UB=[200 0.15 ];

%随机在解空间初始化N个烟花位置
x = zeros(N,D);
for i=1:N
    x(i,:)=LB+rand(1,D).*(UB-LB);
end
%循环迭代
E_Spark=zeros(T,D,N);
Fit = zeros(1,N);
F = zeros(1,T);
for t=1:T
    %计算每个烟花适应度值
    for i=1:N
        Fit(i)=fitness(x(i,:));
    end
    [F(t),~]=min(Fit);
    Fmin=min(Fit);
    Fmax=max(Fit);
    %计算每个烟花的爆炸半径E_R和爆炸数目E_N以及产生的爆炸火花
    E_R = zeros(1,N);
    E_N = zeros(1,N);
    for i=1:N
        E_R(i)=Er*((Fit(i)-Fmin+eps)/(sum(Fit)-N*Fmin+eps));  %爆炸半径
        E_N(i)=En*((Fmax-Fit(i)+eps)/(N*Fmax-sum(Fit)+eps));  %爆炸数目
        if E_N(i)<a*En    % 爆炸数目限制
            E_N(i)=round(a*En);
        elseif E_N(i)>b*En
            E_N(i)=round(b*En);
        else
            E_N(i)=round(E_N(i));
        end
        %产生爆炸火花 E_Spark
        for j=2:(E_N(i)+1)              % 第i个烟花共产生E_N(i)个火花
            E_Spark(1,:,i)=x(i,:);      % 将第i个烟花保存为第i个火花序列中的第一个，爆炸产生的火花从序列中的第二个开始存储（即烟花为三维数组每一页的第一行）
            h=E_R(i)*(-1+2*rand(1,D));  % 位置偏移
            E_Spark(j,:,i)=x(i,:)+h;    % 第i个烟花（三维数组的i页）产生的第j（三维数组的j行）个火花
            for k=1:D   %越界检测
                if E_Spark(j,k,i)>UB(k)||E_Spark(j,k,i)<LB(k)  %第i个烟花（三维数组的i页）产生的第j个火花（三维数组的j行）的第k个变量（三维数组的k列）
                    E_Spark(j,k,i)=LB(k)+rand*(UB(k)-LB(k));   %映射规则
                end
            end
        end
    end
    %产生高斯变异火花Mut_Spark,随机选择M个烟花进行变异
    Mut=randperm(N);          % 随机产生1-N内的N个数
    for m1=1:M                % M个变异烟花
        m=Mut(m1);            % 随机选取烟花
        for n=1:E_N(m)
            e=1+sqrt(1)*randn(1,D); %高斯变异参数，方差为1，均值也为1的1*D随机矩阵
            E_Spark(n,:,m)=E_Spark(n,:,m).*e;
            for k=1:D   %越界检测
                if E_Spark(n,k,m)>UB(k)||E_Spark(n,k,m)<LB(k)  %第i个烟花（三维数组的i页）产生的第j个火花（三维数组的j行）的第k个变量（三维数组的k列）
                    E_Spark(n,k,m)=LB(k)+rand*(UB(k)-LB(k));   %映射规则
                end
            end
        end
    end
    
    %选择操作，从烟花、爆炸火花、变异火花里（都包含在三维数组中）选取N个优良个体作为下一代（先将最优个体留下，然后剩下的N-1个按轮盘赌原则选取）
    n=sum(E_N)+N; %烟花、火花总个数
    q=1;
    Fitness = zeros(1,1);
    E_Sum = zeros(1,D);
    for i=1:N  % 三维转二维
        for j=1:(E_N(i)+1)  % 三维数组每一页的行数（即每个烟花及其产生的火花数之和）
            E_Sum(q,:)=E_Spark(j,:,i); % 烟花与火花总量
            Fitness(q)=fitness(E_Sum(q,:)); % 计算所有烟花、火花的适应度，用于选取最优个体
            q=q+1;
        end
    end
    [Fitness,X]=sort(Fitness);  % 适应度升序排列
    x(1,:)=E_Sum(X(1),:);    % 最优个体
    dist=pdist(E_Sum);       % 求解各火花两两间的欧式距离
    S=squareform(dist);      % 将距离向量重排成n*n数组，第i行之和即为第i个火花到其他火花的距离之和
    P = zeros(1,n);
    for i=1:n                % 分别求各行之和
        P(i)=sum(S(i,:));
    end
    [P,Ix]=sort(P,'descend');% 将距离按降序排列，选取前N-1个，指的是如果个体密度较高，即该个体周围有很多其他候选者个体时，该个体被选择的概率会降低
    for i=1:(N-1)
        x(i+1,:)=E_Sum(Ix(i),:);
    end
end

for i=1:N
    Fit(i)=fitness(x(i,:));
end

[F(T),Y]=min(Fit);
g1=x(Y,:);
zhongjian1_num = round(g1(1));  
xue = g1(2);
%% 模型建立与训练
layers = [ ...
    sequenceInputLayer(shuru_num)    % 输入层
    lstmLayer(zhongjian1_num)        % LSTM层
    fullyConnectedLayer(shuchu_num)  % 全连接层
    regressionLayer];
 
options = trainingOptions('adam', ...   % 梯度下降
    'MaxEpochs',50, ...                % 最大迭代次数
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




















