clear
close  all
%% ���ݶ�ȡ
geshu=500%ѵ�����ĸ���
%��ȡ����
shuru=xlsread('���ݵ�����.csv');
shuchu=xlsread('���ݵ����.csv');
%nn = randperm(size(shuru,1));%�������
nn=1:size(shuru,1);%��������
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
%% �����㷨ѡ����ѵ�BP���� 
N = 10; %����Ѩ����(��������Ĺ�ģ)
D = 2; % ���ά��
T = 50 ; % ��������
% �Ա���������
ParticleScope(1,:)=[10 200];
ParticleScope(2,:)=[0.01 0.15];
ParticleScope=ParticleScope';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xv=rand(N,D); %���ȣ���ʼ����Ⱥ�����ٶȺ�λ��
for d=1:D
    xv(:,d)=xv(:,d)*(ParticleScope(2,d)-ParticleScope(1,d))+ParticleScope(1,d);  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nest_Pop=xv(:,1:D);  % �����ʼ�������
Pa = 0.15 ; % ���³��ĸ���(�������������񵰺�)
for t=1:T
    levNestPop =  func_levy(Nest_Pop,ParticleScope) ;
    % �����µĽ�    
    Nest_Pop = func_bestNestPop(Nest_Pop,levNestPop);  
    % ���³��;ɳ���ѡ��һ����õĳ� 
    rand_nestPop = func_newBuildNest(Nest_Pop,Pa,ParticleScope);
    % ͨ��ƫ��������߷��������ĳ��������µĳ�
    Nest_Pop = func_bestNestPop(Nest_Pop,rand_nestPop) ; 
    % ���³��;ɳ���ѡ��һ����õĳ�
    [~,index] = max(func_fitness(Nest_Pop)) ;
    % ��õĳ�Ѩ   
    trace(t) = func_objValue(Nest_Pop(index,:)) ; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ����ѡ������ֵ
x=Nest_Pop(index,:);
zhongjian1_num = round(x(1));  
xue = x(2);
%% ģ�ͽ�����ѵ��
layers = [ ...
    sequenceInputLayer(shuru_num)    % �����
    lstmLayer(zhongjian1_num)        % LSTM��
    fullyConnectedLayer(shuchu_num)  % ȫ���Ӳ�
    regressionLayer];
 
options = trainingOptions('adam', ...   % �ݶ��½�
    'MaxEpochs',80, ...                % ����������
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




















