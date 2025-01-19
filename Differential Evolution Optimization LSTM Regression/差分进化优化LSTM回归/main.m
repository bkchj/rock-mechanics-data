clear
close  all
%% ���ݶ�ȡ
geshu=950;%ѵ�����ĸ���
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

%% ���ò�ֽ����㷨
D=2;%��������%������ά��
NP=5;                                %������Ŀ                               
G=10;                                %����������
F=0.5;                                %��������
CR=0.5;                               %��������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xs=[300, 0.15]';                                 %����
Xx=[50, 0.01]';                                %����
%%%%%%%%%%%%%%%%%%%%%%%%%����ֵ%%%%%%%%%%%%%%%%%%%%%%%%   
xx=zeros(D,NP);                        %��ʼ��Ⱥ
v=zeros(D,NP);                        %������Ⱥ
u=zeros(D,NP);                        %ѡ����Ⱥ
xchu=rand(D,NP);
for i=1:NP
xx(:,i)=xchu(:,i).*(Xs-Xx)+Xx;              %����ʼ��Ⱥ��ֵ
end
%%%%%%%%%%%%%%%%%%%%����Ŀ�꺯��%%%%%%%%%%%%%%%%%%%%%%%
for m=1:NP
    Ob(m)=fitness(xx(:,m));
end
trace(1)=min(Ob);
%%%%%%%%%%%%%%%%%%%%%%%��ֽ���ѭ��%%%%%%%%%%%%%%%%%%%%%
for gen=1:G
    %%%%%%%%%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%r1,r2,r3��m������ͬ%%%%%%%%%%%%%%%
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
    %%%%%%%%%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%%%%%%%%%%
    r=randi([1,D],1,1);
    for n=1:D
        cr=rand(1);
        if (cr<=CR)||(n==r)
            u(n,:)=v(n,:);
        else
            u(n,:)=xx(n,:);
        end
    end
    %%%%%%%%%%%%%%%%%%%�߽������Ĵ���%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%�߽�����%%%%%%%%%%%%%%%%%%%%%%%%%
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
    %%%%%%%%%%%%%%%%%%%%%%ѡ�����%%%%%%%%%%%%%%%%%%%%%%%
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
X=xx(:,1);                              %���ű���
c = X(1);  
g = X(2); 
toc
zhongjian1_num = ceil(c);  
xue = g;
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
% rmse = sqrt(mean((test_simu-output_train).^2));   % ѵ����
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




















