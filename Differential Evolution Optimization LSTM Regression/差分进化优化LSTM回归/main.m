clear
close  all
%% 数据读取
geshu=950;%训练集的个数
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

%% 利用差分进化算法
D=2;%变量个数%变量的维数
NP=5;                                %个体数目                               
G=10;                                %最大进化代数
F=0.5;                                %变异算子
CR=0.5;                               %交叉算子
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xs=[300, 0.15]';                                 %上限
Xx=[50, 0.01]';                                %下限
%%%%%%%%%%%%%%%%%%%%%%%%%赋初值%%%%%%%%%%%%%%%%%%%%%%%%   
xx=zeros(D,NP);                        %初始种群
v=zeros(D,NP);                        %变异种群
u=zeros(D,NP);                        %选择种群
xchu=rand(D,NP);
for i=1:NP
xx(:,i)=xchu(:,i).*(Xs-Xx)+Xx;              %赋初始种群初值
end
%%%%%%%%%%%%%%%%%%%%计算目标函数%%%%%%%%%%%%%%%%%%%%%%%
for m=1:NP
    Ob(m)=fitness(xx(:,m));
end
trace(1)=min(Ob);
%%%%%%%%%%%%%%%%%%%%%%%差分进化循环%%%%%%%%%%%%%%%%%%%%%
for gen=1:G
    %%%%%%%%%%%%%%%%%%%%%%变异操作%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%r1,r2,r3和m互不相同%%%%%%%%%%%%%%%
    for m=1:NP
        r1=randi([1,NP],1,1);
        while (r1==m)
            r1=randi([1,NP],1,1);
        end
        r2=randi([1,NP],1,1);
        while (r2==m)||(r2==r1)
            r2=randi([1,NP],1,1);
        end
        r3=randi([1,NP],1,1);
        while (r3==m)||(r3==r1)||(r3==r2)
            r3=randi([1,NP],1,1);
        end
        v(:,m)=xx(:,r1)+F*(xx(:,r2)-xx(:,r3));
    end
    %%%%%%%%%%%%%%%%%%%%%%交叉操作%%%%%%%%%%%%%%%%%%%%%%%
    r=randi([1,D],1,1);
    for n=1:D
        cr=rand(1);
        if (cr<=CR)||(n==r)
            u(n,:)=v(n,:);
        else
            u(n,:)=xx(n,:);
        end
    end
    %%%%%%%%%%%%%%%%%%%边界条件的处理%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%边界吸收%%%%%%%%%%%%%%%%%%%%%%%%%
    for n=1:D
        for m=1:NP
            if u(n,m)<Xx(n)
                u(n,m)=Xx(n);
            end
            if u(n,m)>Xs(n)
                u(n,m)=Xs(n);
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%选择操作%%%%%%%%%%%%%%%%%%%%%%%
    for m=1:NP
        Ob1(m)=fitness(u(:,m));
    end
    for m=1:NP
        if Ob1(m)<Ob(m)
            xx(:,m)=u(:,m);
        end
    end
    for m=1:NP
        Ob(m)=fitness(xx(:,m));
    end
    trace(gen+1)=min(Ob);
end
[SortOb,Index]=sort(Ob);
xx=xx(:,Index);
X=xx(:,1);                              %最优变量
c = X(1);  
g = X(2); 
toc
zhongjian1_num = ceil(c);  
xue = g;
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
% rmse = sqrt(mean((test_simu-output_train).^2));   % 训练集
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




















