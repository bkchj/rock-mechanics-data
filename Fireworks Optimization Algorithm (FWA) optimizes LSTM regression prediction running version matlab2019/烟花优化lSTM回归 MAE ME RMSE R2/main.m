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
N=5;   % N�̻���
T=10;   % TΪ��������
D=2;     % D����ά��


M=5;     % M�������
En=6;    % En��ը��Ŀ
Er=5;    % Er��ը�뾶
a=0.3;   % a,bΪ��ը��Ŀ��������
b=0.6;

%�����ֵ�������½�
LB=[10 0.01 ];
UB=[200 0.15 ];

%����ڽ�ռ��ʼ��N���̻�λ��
x = zeros(N,D);
for i=1:N
    x(i,:)=LB+rand(1,D).*(UB-LB);
end
%ѭ������
E_Spark=zeros(T,D,N);
Fit = zeros(1,N);
F = zeros(1,T);
for t=1:T
    %����ÿ���̻���Ӧ��ֵ
    for i=1:N
        Fit(i)=fitness(x(i,:));
    end
    [F(t),~]=min(Fit);
    Fmin=min(Fit);
    Fmax=max(Fit);
    %����ÿ���̻��ı�ը�뾶E_R�ͱ�ը��ĿE_N�Լ������ı�ը��
    E_R = zeros(1,N);
    E_N = zeros(1,N);
    for i=1:N
        E_R(i)=Er*((Fit(i)-Fmin+eps)/(sum(Fit)-N*Fmin+eps));  %��ը�뾶
        E_N(i)=En*((Fmax-Fit(i)+eps)/(N*Fmax-sum(Fit)+eps));  %��ը��Ŀ
        if E_N(i)<a*En    % ��ը��Ŀ����
            E_N(i)=round(a*En);
        elseif E_N(i)>b*En
            E_N(i)=round(b*En);
        else
            E_N(i)=round(E_N(i));
        end
        %������ը�� E_Spark
        for j=2:(E_N(i)+1)              % ��i���̻�������E_N(i)����
            E_Spark(1,:,i)=x(i,:);      % ����i���̻�����Ϊ��i���������еĵ�һ������ը�����Ļ𻨴������еĵڶ�����ʼ�洢�����̻�Ϊ��ά����ÿһҳ�ĵ�һ�У�
            h=E_R(i)*(-1+2*rand(1,D));  % λ��ƫ��
            E_Spark(j,:,i)=x(i,:)+h;    % ��i���̻�����ά�����iҳ�������ĵ�j����ά�����j�У�����
            for k=1:D   %Խ����
                if E_Spark(j,k,i)>UB(k)||E_Spark(j,k,i)<LB(k)  %��i���̻�����ά�����iҳ�������ĵ�j���𻨣���ά�����j�У��ĵ�k����������ά�����k�У�
                    E_Spark(j,k,i)=LB(k)+rand*(UB(k)-LB(k));   %ӳ�����
                end
            end
        end
    end
    %������˹�����Mut_Spark,���ѡ��M���̻����б���
    Mut=randperm(N);          % �������1-N�ڵ�N����
    for m1=1:M                % M�������̻�
        m=Mut(m1);            % ���ѡȡ�̻�
        for n=1:E_N(m)
            e=1+sqrt(1)*randn(1,D); %��˹�������������Ϊ1����ֵҲΪ1��1*D�������
            E_Spark(n,:,m)=E_Spark(n,:,m).*e;
            for k=1:D   %Խ����
                if E_Spark(n,k,m)>UB(k)||E_Spark(n,k,m)<LB(k)  %��i���̻�����ά�����iҳ�������ĵ�j���𻨣���ά�����j�У��ĵ�k����������ά�����k�У�
                    E_Spark(n,k,m)=LB(k)+rand*(UB(k)-LB(k));   %ӳ�����
                end
            end
        end
    end
    
    %ѡ����������̻�����ը�𻨡�����������������ά�����У�ѡȡN������������Ϊ��һ�����Ƚ����Ÿ������£�Ȼ��ʣ�µ�N-1�������̶�ԭ��ѡȡ��
    n=sum(E_N)+N; %�̻������ܸ���
    q=1;
    Fitness = zeros(1,1);
    E_Sum = zeros(1,D);
    for i=1:N  % ��άת��ά
        for j=1:(E_N(i)+1)  % ��ά����ÿһҳ����������ÿ���̻���������Ļ���֮�ͣ�
            E_Sum(q,:)=E_Spark(j,:,i); % �̻��������
            Fitness(q)=fitness(E_Sum(q,:)); % ���������̻����𻨵���Ӧ�ȣ�����ѡȡ���Ÿ���
            q=q+1;
        end
    end
    [Fitness,X]=sort(Fitness);  % ��Ӧ����������
    x(1,:)=E_Sum(X(1),:);    % ���Ÿ���
    dist=pdist(E_Sum);       % �������������ŷʽ����
    S=squareform(dist);      % �������������ų�n*n���飬��i��֮�ͼ�Ϊ��i���𻨵������𻨵ľ���֮��
    P = zeros(1,n);
    for i=1:n                % �ֱ������֮��
        P(i)=sum(S(i,:));
    end
    [P,Ix]=sort(P,'descend');% �����밴�������У�ѡȡǰN-1����ָ������������ܶȽϸߣ����ø�����Χ�кܶ�������ѡ�߸���ʱ���ø��屻ѡ��ĸ��ʻή��
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
%% ģ�ͽ�����ѵ��
layers = [ ...
    sequenceInputLayer(shuru_num)    % �����
    lstmLayer(zhongjian1_num)        % LSTM��
    fullyConnectedLayer(shuchu_num)  % ȫ���Ӳ�
    regressionLayer];
 
options = trainingOptions('adam', ...   % �ݶ��½�
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




















