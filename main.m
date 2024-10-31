clear all
clc
%% 
T_f = 20;
% 步长
dt = 0.01;
% 仿真步数
steps = T_f/dt;
% 随机状态
randn('state',1);
%% 通信拓扑
% 参考Antagonistic Interaction-Based Bipartite Consensus
% Control for Heterogeneous Networked Systems
% 跟随者个数
N = 5;
% 领导者个数
M = 1;
% 网络拓扑参数
A = [0 1 -1 0 1 1;
    1 0 0 0 1 1; 
    -1 0 0 1 0 0;
    0 0 1 0 0 0;
    1 1 0 0 0 0;
    0 0 0 0 0 0];
D = diag([4 3 2 1 2 0]);
b = [1 1 -1 -1 1];
L = D-A;
L1 = L(1:N,1:N);
L2 = L(1:N,N+1);
phi = [1 1 -1 -1 1];
Phi = diag(phi);
L_r = -inv(L1)*Phi*L2;
%% 初始化
y_l = zeros(M,steps);
dot_y_l = zeros(M,steps);
y_r = zeros(N,steps);
dot_yr = zeros(N,steps);
x_1 = zeros(N,steps);
x_2 = zeros(N,steps);
u = zeros(N,steps);
mu_1 = zeros(N,steps);
% 初始状态1
x_1_0 = [0.2,0.1,-0.2,-0.1,-0.1]';
% 初始状态2
% x_1_0 = [-0.2,0.3,-0.1,0.1,0.2]';
% 初始状态3
% x_1_0 = [-0.3,0.2,-0.3,0.3,0.1]';
x_1(:,1) = x_1_0;
% 权重初始化
vartheta = zeros(N,steps);
vartheta(:,1) = 1*ones(N,1);
% 误差
zeta_1 = zeros(N,steps);
zeta_2 = zeros(N,steps);
e = zeros(N,steps);
trans_e = zeros(N,steps);
rho = zeros(1,steps);

bipartite_error = zeros(N,steps);
bipartite_error2= zeros(N,steps);
norm_bipartite_error = zeros(1,steps);
norm_bipartite_error2 = zeros(1,steps);
% 控制器
alpha_1 = zeros(N,steps);
%% 指定性能参数
% T_s = 1;
% sigma = 0.2;
%% 控制器参数
parameters;

% 不同参数对比实验
para = parameters_main;
T_s = para.T_s;
sigma = para.sigma;
c1 = para.c1;
bar_c1 = para.bar_c1;
c2 =para.c2;
bar_c2 = para.bar_c2;
beta = para.beta;
bar_beta = para.bar_beta;

kappa=1;
epsilon = 1;
lambda = 1;
bar_lambda = 1;
%% 仿真循环
t = 0:dt:T_f;
for i = 1:steps-1
    %% 当前时间
    current_t = dt*i;
    %% 参考信号
    y_l(i) = sin(current_t);
    dot_y_l(i) = cos(current_t);
    
    % 每个智能体的本地参考信号
    y_r(:,i) = L_r*y_l(i);
    dot_yr(:,i) = L_r*dot_y_l(i);
    %% 误差变换
    e(:,i) = x_1(:,i)-y_r(:,i);
    rho = scaling_function(current_t,T_s,sigma);
    trans_e(:,i) = transformation_function(e(:,i),rho);
    mu_1(:,i) = mu(rho,e(:,i));
    zeta_1(:,i) = trans_e(:,i);
    % 同步误差计算
    bipartite_error(:,i) = synchronization_birartite(A,b,x_1(:,i),y_l(:,i));
    norm_bipartite_error(i) = norm(bipartite_error(:,i));
    bipartite_error2(:,i) = L1*e(:,i);
    norm_bipartite_error2(i) = norm(bipartite_error2(:,i));
    %% Step1
%     X_1 = [x_1(:,i),y_r(:,i),dot_yr(:,i),rho.*ones(N,1),vartheta(:,i)];
    X_1 = [x_1(:,i),y_r(:,i),dot_yr(:,i),vartheta(:,i)];
    Psi_Psi = zeros(N,1);
    for j = 1:N
        Psi_1 = FLS_phi(X_1(j,:));
        Psi_Psi(j) = Psi_1*Psi_1';
    end
    alpha_1(:,i) = -(c1.*zeta_1(:,i).^(4*beta-3)+bar_c1.*zeta_1(:,i).^(4*bar_beta-3)+0.5.*zeta_1(:,i).^3-3/4.*zeta_1(:,i)+zeta_1(:,i).^3.*vartheta(:,i)./(4*epsilon.*Psi_Psi)).*mu_1(:,i);
%     alpha_1(:,i) = -(c1.*zeta_1(:,i).^(4*beta-3)+0.5.*zeta_1(:,i).^3-3/4.*zeta_1(:,i)+zeta_1(:,i).^3.*vartheta(:,i)./(4*epsilon.*Psi_Psi)).*mu_1(:,i);
    %% Step 2
    zeta_2(:,i) = x_2(:,i)-alpha_1(:,i);
%     X_2 = [x_1(:,i),x_2(:,i),y_r(:,i),dot_yr(:,i),rho.*ones(N,1),vartheta(:,i)];
    X_2 = [x_1(:,i),x_2(:,i),y_r(:,i),dot_yr(:,i),vartheta(:,i)];
    Psi_Psi_2 = zeros(N,1);
    for j = 1:N
        Psi_2 = FLS_phi(X_2(j,:));
        Psi_Psi_2(j) = Psi_2*Psi_2';
    end
    u(:,i) = -c2.*zeta_2(:,i).^(4*beta-3)-(bar_c2+1).*zeta_2(:,i).^(4*bar_beta-3)-zeta_2(:,i).^3.*vartheta(:,i)./(4*epsilon.*Psi_Psi_2);
%     u(:,i) = -c2.*zeta_2(:,i).^(4*beta-3)-zeta_2(:,i).^3.*vartheta(:,i)./(4*epsilon.*Psi_Psi_2);
    %% 更新权重
%     vartheta(:,i+1)=vartheta(:,i) + ((zeta_1(:,i).^3+zeta_2(:,i).^3).*vartheta(:,i)./(4*epsilon.*Psi_Psi_2)-lambda*vartheta(:,i).^(2*beta-1))*dt;
    vartheta(:,i+1)=vartheta(:,i) + (kappa*(zeta_1(:,i).^3./(4*epsilon.*Psi_Psi)+zeta_2(:,i).^3./(4*epsilon.*Psi_Psi_2))-lambda*vartheta(:,i)-bar_lambda*vartheta(:,i).^(2*beta-1))*dt;
    %% 更新系统状态
    rnd_1 = rand(N,1);
    x_1(:,i+1) = x_1(:,i) + (x_2(:,i)+(1-sin(x_1(:,i)).^2).*x_1(:,i))*dt+0.5*cos(x_1(:,i)).*sqrt(dt).*rnd_1*0.02;
    rnd_2 = rand(N,1);
    x_2(:,i+1) = x_2(:,i) + (u(:,i)+(-3.5.*x_2(:,i)+x_1(:,i).*x_2(:,i).^2))*dt+0.1*x_1(:,i).*sin(2*x_1(:,i).*x_2(:,i)).*sqrt(dt).*rnd_2*0.02;
end
