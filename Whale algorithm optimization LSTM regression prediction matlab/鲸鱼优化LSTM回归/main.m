clear
close  all
%% ���ݶ�ȡ
geshu=900;%ѵ�����ĸ���
%��ȡ����
shuru=xlsread('���ݵ�����.csv');
shuchu=xlsread('���ݵ����.csv');
nn = randperm(size(shuru,1));%�������
% nn=1:size(shuru,1);%��������
input_train =shuru(nn(1:geshu),:);
input_train=input_train';
output_train=shuchu(nn(1:geshu),:);
output_train=output_train';
input_test =shuru(nn((geshu+1):end),:);
input_test=input_test';
output_test=shuchu(nn((geshu+1):end),:);
output_test=output_test';
%��������������ݹ�һ��
[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
global inputn outputn shuru_num shuchu_num
[inputn,inputps]=mapminmax('apply',input_train,bb);
[outputn,outputps]=mapminmax('apply',output_train,dd);
shuru_num = size(input_train,1); % ����ά��
shuchu_num = 1;  % ���ά��
%%
% ������ʼ��
dim=2;
Max_iteration=50;   % ��������  
pop=12;  %��Ⱥ��ģ
lb = [300, 0.15];
ub = [50, 0.01];
fobj = @(x) fun(x);
[Leader_score,Leader_pos,Convergence_curve]=WOA(pop,Max_iteration,lb,ub,dim,fobj); %��ʼ�Ż�

g1=Leader_pos;
zhongjian1_num = round(g1(1));  
xue = g1(2);
%% ģ�ͽ�����ѵ��
layers = [ ...
    sequenceInputLayer(shuru_num)    % �����
    lstmLayer(zhongjian1_num)        % LSTM��
    fullyConnectedLayer(shuchu_num)  % ȫ���Ӳ�
    regressionLayer];
 
options = trainingOptions('SGD', ...   % �ݶ��½�
    'MaxEpochs',50, ...                % ����������
    'GradientThreshold',1, ...         % �ݶ���ֵ 
    'InitialLearnRate',xue,...
    'Verbose',0, ...
    'Plots','training-progress');            % ѧϰ��
%% ѵ��LSTM
net = trainNetwork(inputn,outputn,layers,options);
%% Ԥ��
net = resetState(net);% ����ĸ���״̬���ܶԷ�������˸���Ӱ�졣��������״̬���ٴ�Ԥ�����С�
[~,Ytrain]= predictAndUpdateState(net,inputn);
test_simu=mapminmax('reverse',Ytrain,dd);%����һ��
rmse = sqrt(mean((test_simu-output_train).^2));   % ѵ����
%���Լ���������������ݹ�һ��
inputn_test=mapminmax('apply',input_test,bb);
[net,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%����һ��
error1=test_simu1-output_test;%���Լ�Ԥ��-��ʵ
%������������ (RMSE)��
rmse1 = sqrt(mean((test_simu1-output_test).^2));  % ���Լ�
%% ��ͼ

%��Ԥ��ֵ��������ݽ��бȽϡ�
figure
plot(output_train)
hold on
plot(test_simu,'.-')
hold off
legend(["��ʵֵ" "Ԥ��ֵ"])
xlabel("����")
title("ѵ����")


figure
plot(output_test)
hold on
plot(test_simu1,'.-')
hold off
legend(["��ʵֵ" "Ԥ��ֵ"])
xlabel("����")
title("���Լ�")


 % ��ʵ���ݣ�������������������������������output_test = output_test;
T_sim_optimized = test_simu1;  % ��������

num=size(output_test,2);%ͳ����������
error=T_sim_optimized-output_test;  %�������
mae=sum(abs(error))/num; %����ƽ���������
me=sum((error))/num; %����ƽ���������
mse=sum(error.*error)/num;  %����������
rmse=sqrt(mse);     %�����������
% R2=r*r;
tn_sim = T_sim_optimized';
tn_test =output_test';
N = size(tn_test,1);
R2=(N*sum(tn_sim.*tn_test)-sum(tn_sim)*sum(tn_test))^2/((N*sum((tn_sim).^2)-(sum(tn_sim))^2)*(N*sum((tn_test).^2)-(sum(tn_test))^2)); 

disp(' ')
disp('----------------------------------------------------------')

disp(['ƽ���������maeΪ��            ',num2str(mae)])
disp(['ƽ�����meΪ��            ',num2str(me)])
disp(['��������rmseΪ��             ',num2str(rmse)])
disp(['���ϵ��R2Ϊ��                ' ,num2str(R2)])




















