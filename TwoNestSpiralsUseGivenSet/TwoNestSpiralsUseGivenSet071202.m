
% 可以自由设置螺旋曲线的样本个数。 可以正确画出正负类的区域。
% 可以使用softmax
% 加入正则项
% 给隐藏层最后一层使用dropout
clear;
%% 产生双螺旋数据
train_num=100;  % 0和1的样本各train_num个。 可以设置任意数目。
% 
train_i=(1: (105-1)/train_num: 105-(105-1)/train_num)';

%双螺旋数据点的产生方程
alpha1=pi*(train_i-1)/25;
beta=0.4*((105-train_i)/(104)); %
x0=beta.*cos(alpha1);
y0=beta.*sin(alpha1);
z0=zeros(train_num,1);
x1=-beta.*cos(alpha1);
y1=-beta.*sin(alpha1);
z1=ones(train_num,1);

% 随机打乱顺序
k=rand(1,2*train_num);
[m,n]=sort(k);

train=[x0 y0 z0;x1,y1,z1]; 
train_label1=train(n(1:2*train_num),end)';    % 1*(2*train_num)
train_data1=train(n(1:2*train_num),1:end-1)'; % 2*(2*train_num)
      
% 把1维的输出变成2维的输出 0->[1 0], 1->[0 1]
for i=1:2*train_num
    switch train_label1(i)
        case 0
            train_label2(i,:)=[1 0];
        case 1
            train_label2(i,:)=[0 1];
    end
end

train_label=train_label2'; % 2*200
% 均值归一化
[train_data,train_datas]=mapminmax(train_data1);

% 在plot上显示训练样本
plot(x0,y0,'r+');
hold on;
plot(x1,y1,'go');

%legend();

%% BP神经网络结构的初始化
% 参数设定
train_time=45000;
alpha=0.003;
lambda = 0.1;
vbeta = 0.9;
% dropout率
keep_prob = 0.9;
% 网络结构
innum=2;
midnum1=12;
midnum2=12;
midnum3=12;
outnum=2;

% 初始化权重和偏置
w1= randn(midnum1,innum);
b1=0;
w2= randn(midnum2,midnum1);
b2=0;
w3= randn(midnum3,midnum2);
b3=0;
w4= randn(outnum,midnum3);
b4=0;

errors=[];

% 初始化动量因子
vdw1=0;
vdb1=0;
vdw2=0;
vdb2=0;
vdw3=0;
vdb3=0;
vdw4=0;
vdb4=0;

%% 训练model
for train_circle=1:train_time

    z1 = w1 * train_data + b1;  % 12*(2*train_num)
    a1 = ReLU(z1);              % 12*(2*train_num) 
    z2 = w2 * a1 + b2; 
    a2 = ReLU(z2);              % 2*(2*train_num)
    % 分类问题用dropout，在最后一层softmax前用基本就可以了，能够防止过拟合。
    % 产生（0，1）随机数，如果大于keep_prob则为1，反之为0。
    dropout3 = rand(midnum3,1)<keep_prob;
    z3 = w3 * a2 + b3;          % 2*(2*train_num)
    a3 = ReLU(z3) .* dropout3;  % 2*(2*train_num)
    a3 = a3 ./ keep_prob;
    z4 = w4 * a3 + b4;          % 2*(2*train_num)
    % softmax
    a4=softmax(z4);
    
    %% 计算误差
    loss = -train_label .* log(a4); 
    J = sum((1 / (2 * train_num)) * sum(loss));
    reg  = (lambda / (2 * train_num)) * (sum(sum(w1.^2,2)) + sum(sum(w2.^2,2)) + sum(sum(w3.^2,2)) + sum(sum(w4.^2,2))); 
    J = J + reg;
    
    errors = [errors; J];
    
    % 计算 梯度   
    % 正则化项
    regDw4 = (lambda / (2 * train_num)) * w4;
    regDb4 = (lambda / (2 * train_num)) * b4;
    regDw3 = (lambda / (2 * train_num)) * w3;
    regDb3 = (lambda / (2 * train_num)) * b3;
    regDw2 = (lambda / (2 * train_num)) * w2;
    regDb2 = (lambda / (2 * train_num)) * b2;
    regDw1 = (lambda / (2 * train_num)) * w1;
    regDb1 = (lambda / (2 * train_num)) * b1;
    
    % da4 = -1 ./ a4 ;   % 算不算da4无所谓，因为求dz4的时候，已经消掉了一个a4
    % 需要加负号，否则会趋向于无穷大
    dz4 = a4 - train_label;                       % 2*(2*train_num)
    dw4 = dz4 * a3' ./ (2 * train_num) - regDw4;     % 2*12
    db4 = mean(dz4, 2) - regDb4;                     % 2*1
    
    % dropout 将隐藏层的第三层进行dropout 
    da3 = w4' * dz4;                     % 12*(2*train_num)
    da3 = da3 .* dropout3;
    da3 = da3 ./ keep_prob;
    dz3 = da3 .* ReLUGradient(z3);                   % 12*(2*train_num)
    dw3 = dz3 * a2' ./ (2 * train_num) + regDw3;     % 12*12
    db3 = mean(dz3, 2) - regDb3;                     % 12*1

    da2 = w3' * dz3 ;                                % 12*(2*train_num)
    dz2 = da2 .* ReLUGradient(z2);                   % 2*(2*train_num)
    dw2 = dz2 * a1' ./ (2 * train_num) + regDw2;     % 12*12
    db2 = mean(dz2, 2) - regDb2;                     % 12*1
    
    da1 = w2' * dz2;                                      % 12*(2*train_num)
    dz1 = da1 .* ReLUGradient(z1);                        % 2*(2*train_num)
    dw1 = dz1 * train_data' ./ (2 * train_num) + regDw1;  % 12*2
    db1 = mean(dz1, 2) - regDb1;                          % 12*1

    % 使用动量（momentum）。当本次梯度方向与上一次相同时，正向加速。方向相反时，减速。
    vdw1 = vbeta * vdw1 + dw1;
    vdb1 = vbeta * vdb1 + db1;
    vdw2 = vbeta * vdw2 + dw2;
    vdb2 = vbeta * vdb2 + db2;
    vdw3 = vbeta * vdw3 + dw3;
    vdb3 = vbeta * vdb3 + db3;
    vdw4 = vbeta * vdw4 + dw4;
    vdb4 = vbeta * vdb4 + db4;
    
   % 权值更新方程
    w1 = w1 - alpha * vdw1; 
    b1 = b1 - alpha * vdb1; 
    w2 = w2 - alpha * vdw2; 
    b2 = b2 - alpha * vdb2; 
    w3 = w3 - alpha * vdw3; 
    b3 = b3 - alpha * vdb3; 
    w4 = w4 - alpha * vdw4; 
    b4 = b4 - alpha * vdb4;    
end

%% 产生双螺旋测试数据
test_num = 100;

%% 产生双螺旋数据,每类100个样本点，共200个样本
test_i=(1.5: (105-1.5)/test_num : 105-(105-1.5)/test_num)';    %每类51个样本

%双螺旋数据点的产生方程
alpha2=pi*(test_i-1)/25;
beta2=0.4*((105-test_i)/104);
m0=beta2.*cos(alpha2);
n0=beta2.*sin(alpha2);
s0=zeros(test_num,1);
m1=-beta2.*cos(alpha2);
n1=-beta2.*sin(alpha2);
s1=ones(test_num,1);

test=[m0 n0 s0;m1,n1,s1]; %1条螺旋线数据点,3*102的矩阵
test_label1=test(:,end)';   %测试数据类别，1*102的行向量
test_data1=test(:,1:end-1)'; %测试数据属性，2*102的矩阵

%把1维的输出变成2维的输出
for i=1:2*test_num
    switch test_label1(i)
        case 0
            test_label2(i,:)=[1 0];
        case 1
            test_label2(i,:)=[0 1];
    end
end

test_label=test_label2'; 

%  画出测试数据双螺旋曲线
% plot(m0,n0,'k^');
% hold on;
% plot(m1,n1,'ms');
% legend('训练数据螺旋线1','训练数据螺旋线2','测试数据螺旋线1','测试数据螺旋线2');
% 归一化函数mapminmax,其可以归一化到[-1,1]或者[0,1]。
test_data=mapminmax('apply',test_data1,train_datas);

%% 预测数据label
% test_z1=w1*train_data+b1;  % 使用训练样本进行预测
test_z1=w1*test_data+b1; 
test_a1=ReLU(test_z1); 

test_z2=w2*test_a1+b2;
test_a2=ReLU(test_z2); 

test_z3=w3*test_a2+b3;
test_a3=ReLU(test_z3); 

test_z4=w4*test_a3+b4;
predict = softmax(test_z4);

%% 预测结果分析
% output_pred 为1 代表预测为[1 0]，为2代表预测为[0 1].   0->[1 0], 1->[0 1]
for i=1:2*test_num
% for i=1:2*train_num
    output_pred(i)=find(predict(:,i)==max(predict(:,i)));
end
% 0:预测正确；-1：预测为0，label为1；1：预测为1，label为0
error=output_pred-test_label1-1;    %

% 计算出每一类预测错误的个数总和
k=zeros(1,2); %k=[0 0]
for i=1:2*test_num
    if error(i)~=0    %matlab中不能用if error(i)！=0 
        [b c]=max(test_label(:,i));
        switch c
            case 1
                k(1)=k(1)+1;
            case 2
                k(2)=k(2)+1;
        end
    end
end

% 求出每一类总体的个数和
kk=zeros(1,2); %k=[0 0]
for i=1:2*test_num
    [b c]=max(test_label(:,i));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
    end
end

% 计算每一类的正确率
accuracy=(kk-k)./kk

%% 画出该model在二维坐标的label分布。
% 坐标集合 x坐标范围[-0.4,0.4]间隔0.01， y坐标范围[-0.4,0.4]间隔0.01 
coordinateSet = [];
xBegin = -0.4;
xEnd = 0.4;
yBegin = -0.4;
yEnd = 0.4;
interval = 0.01;
% 生成坐标集合
for x= xBegin : interval : xEnd
    for y = yBegin : interval : yEnd
        coordinateSet = [coordinateSet, [x;y] ];
    end;
end;
% 归一化
coordinateSet=mapminmax('apply',coordinateSet,train_datas);

coordinateZ1=w1*coordinateSet+b1;
coordinateA1=ReLU(coordinateZ1); 

coordinateZ2=w2*coordinateA1+b2;
coordinateA2=ReLU(coordinateZ2); 

coordinateZ3=w3*coordinateA2+b3;
coordinateA3=ReLU(coordinateZ3); 

coordinateZ4=w4*coordinateA3+b4;
coordinatePredict = softmax(coordinateZ4);

positiveX = [];
positiveY = [];
negativeX = [];
negativeY = [];

% n为样本的数量
[m,n] = size(coordinatePredict);
for i=1:n
    % coordinate_output_pred 为1 代表预测为[1 0]，为2代表预测为[0 1]
    coordinate_output_pred(i)=find(coordinatePredict(:,i)==max(coordinatePredict(:,i)));    %out_pred为1*102的矩阵
end

coordinaI = 1;
% 根据label，将坐标分到positive集合，或者negative集合。
for x= xBegin : interval : xEnd
    for y = yBegin : interval : yEnd
        if coordinate_output_pred(1,coordinaI) == 1
            negativeX = [negativeX, x ];
            negativeY = [negativeY, y ];
        elseif coordinate_output_pred(1,coordinaI) == 2
            positiveX = [positiveX, x ];
            positiveY = [positiveY, y ];
        end;
        coordinaI=coordinaI+1;
    end;
end;

% 将positive集合 和negative集合显示到图上。
plot(negativeX,negativeY,'c+');
hold on;
plot(positiveX,positiveY,'yo');
legend('训练数据螺旋线1','训练数据螺旋线2','判定为0的区域','判定为1的区域');

%% error curve
figure;
plot(errors);
