clear all 
clc

d = 0.05;
N = 20;
%% Robotic arm model
l1 = 10; % length of first arm
l2 = 7; % length of second arm
theta1_start = 0;
theta1_end = pi/2;
theta2_start = 0;
theta2_end = pi;
theta1 = theta1_start:d:pi/theta1_end; % all possible theta1 values
theta2 = theta2_start:d:theta2_end;
[THETA1,THETA2] = meshgrid(theta1,theta2);
X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2); % compute x coordinates
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2); % training output data
inputData = [THETA1(:),THETA2(:)]';
outputData = [X(:),Y(:)]';
% Umgebung
theta1_u = theta1-1/2*d;
theta1_l = theta1+1/2*d;
theta2_u = theta2-1/2*d;
theta2_l = theta2+1/2*d;
[Theta1_u,Theta2_u]=meshgrid(theta1_u,theta2_u);
[Theta1_l,Theta2_l]=meshgrid(theta1_l,theta2_l);
Input_u = [Theta1_u(:),Theta2_u(:)]';
Input_l = [Theta1_l(:),Theta2_l(:)]';

%
[S,Q] = size(inputData);
[R,Q] = size(outputData);
%% Train ELM with robust optimization method
%1. Randomly Generate the Input Weight Matrix
    IW = rand(N,R) * 2 - 1;
%2. Randomly Generate the Bias Matrix
    B = rand(N,1);
    BiasMatrix = repmat(B,1,Q);
%3. Compute the Layer Output Matrix H
    [H_l,H_u]=computelayerOutput(Input_l',Input_u',IW,B,'sig');
    H=1/2*(H_l+H_u);
    H1=1/2*(H_u-H_l);
%4. RO SDP
   %SDP var
    LW = sdpvar(N,2);
    lamda = sdpvar(2,2);
    tao = sdpvar(2,2);
    M = H1*LW;
    F =(H*LW-outputData');
    l = norm(H*LW-outputData');
   %SDP constrain
    Cons=[tao>=0;[lamda-tao zeros(2,2) F';zeros(2,2) tao M';F M eye(size(M,1))]>=0];
   optimize(Cons,lamda);
   LW=value(LW);

X=inputData;
ELM.weight{1} = IW;
ELM.weight{2} = LW';
ELM.bias{1} = B;
ELM.bias{2} = 0;
ELM.activeFcn = {'sig','purelin'};
Y=elmpredict(X,ELM);
%% ELM
tempH = IW * inputData + BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
% Calculate the Output Weight Matrix
LW1 =pinv(H')*outputData';
ELM1=ELM;
ELM1.weight{2}=LW1';
Y1=elmpredict(X,ELM1);
ERR=LW-LW1;
%% * Output guaranteed distance *
% Output Interval
options.tol = d*0.01;
tol=options.tol;
%Outputset of Robotic arm model 
THEta1 = theta1_start:tol:pi/theta1_end; % all possible theta1 values
THEta2 = theta2_start:tol:theta2_end;
[THETA1,THETA2] = meshgrid(THEta1,THEta2);
X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2); % compute x coordinates
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2); % training output data
outputData = [X(:),Y(:)]';
for i= 1:size(outputData,2)
      for j=1:2
           outputset{1,i}(1,1) = outputData(1,i)-l1*tol;
           outputset{1,i}(1,2) = outputData(1,i)+l1*tol;
           outputset{1,i}(2,1) = outputData(2,i)-l2*tol;
           outputset{1,i}(2,2)=  outputData(2,i)+l2*tol;
     end
end

%Outputset of ELM
inputIntvl=[0,pi/2;0,pi];
elm_ro = ffnetwork(ELM.weight,ELM.bias,ELM.activeFcn);
elm = ffnetwork(ELM1.weight,ELM1.bias,ELM1.activeFcn);
yInterval_ro = outputSet(elm_ro,inputIntvl,options);
yInterval_el = outputSet(elm,inputIntvl,options);
%1.Output mixed method for ELMs
%dis_ro=zeros(size(yInterval_ro,2),1);
%dis_el=zeros(size(yInterval_ro,2),1);
%
distance_ro=zeros(length(yInterval_ro),1);
distance_el=zeros(length(yInterval_el),1);
lossmax_ro=zeros(length(yInterval_ro),2);
lossmax_el=zeros(length(yInterval_ro),2);
for j = 1:length(yInterval_ro)
    for i= 1:2
        lossmax_ro(j,i) = max([outputset{1,j}(i,2)-yInterval_ro{1,j}(i,1),yInterval_ro{1,j}(i,2)-outputset{1,j}(i,1)]);
        lossmax_el(j,i) = max([outputset{1,j}(i,2)-yInterval_el{1,j}(i,1),yInterval_el{1,j}(i,2)-outputset{1,j}(i,1)]);
    end
end
for i=1:length(yInterval_ro)
   distance_ro(i,1)=norm(lossmax_ro(i,:)',2);
   distance_el(i,1)=norm(lossmax_el(i,:)',2);
end
e_max_reach_ro = max(distance_ro);
e_max_reach_el = max(distance_el);
%
inputData = [THETA1(:),THETA2(:)]';
y_el=elmpredict(inputData,ELM1);
y_ro=elmpredict(inputData,ELM);
%
dis_elmro=vecnorm(y_ro-outputData);
dis_elm=vecnorm(y_el-outputData);
dist_sample_ro = max(dis_elmro);
dist_sample_el = max(dis_elm);
%% Lipschitz method 
Lipschitz_ro =dist_sample_ro+(0.25*norm(ELM.weight{1,1},2)*norm(ELM.weight{1,2},2)+sqrt(2)*(l1+l2))*tol;
Lipschitz_el =dist_sample_el+(0.25*norm(ELM1.weight{1,1},2)*norm(ELM1.weight{1,2},2)+sqrt(2)*(l1+l2))*tol;

%% Plot figures 

numPoints = 500;
theta1_rand = (theta1_end-theta1_start).*rand(1,numPoints) + theta1_start;
theta2_rand = (theta2_end-theta2_start).*rand(1,numPoints) + theta2_start;
input_rand = [theta1_rand;theta2_rand]; % random numPoints inputs
output_elm = elmpredict(input_rand,ELM1); % random numPoints outputs
output_roelm=elmpredict(input_rand,ELM);
X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data

figure('NumberTitle', 'off', 'Name', 'Set-valued Reachability: Orignal ELM')
plot(output_elm(1,:),output_elm(2,:),'o')
hold on
plot(X_f,Y_f,'*')
for i = 1:1:numPoints
    center = output_elm(:,i)';
    circle(center,e_max_reach_el,1000,'--');
    hold on
end
plot(output_elm(1,:),output_elm(2,:),'o')
plot(X_f,Y_f,'*')
legend('Actual Position','Predict Position','Error bound')
axis equal

figure('NumberTitle', 'off', 'Name', 'Set-valued Reachability: Robust ELM')
plot(output_roelm(1,:),output_roelm(2,:),'o')
hold on
plot(X_f,Y_f,'*')
for i = 1:1:numPoints
    center = output_roelm(:,i)';
    circle(center,e_max_reach_ro,1000,'--');
    hold on
end
plot(output_roelm(1,:),output_roelm(2,:),'o')
plot(X_f,Y_f,'*')
legend('Actual Position','Predict Position','Error bound')
axis equal


% figure('NumberTitle', 'off', 'Name',['Guaranteed Distance of ELM'])
% for i = 1:1:size(y_el,2)
%     center = y_el(:,i)';
%     circle(center,e_max_reach_el,1000,'--');
%     hold on
% end
% plot(y_el(1,:),y_el(2,:),'o')
% plot(outputData(1,:),outputData(2,:),'*')
% axis equal
% 
% figure('NumberTitle', 'off', 'Name',['Guaranteed Distance of RO_ELM'])
% for i = 1:1:size(y_el,2)
%     center = y_ro(:,i)';
%     circle(center,e_max_reach_ro,1000,'--');
%     hold on
% end
% plot(y_ro(1,:),y_ro(2,:),'o')
% plot(outputData(1,:),outputData(2,:),'*')
% axis equal
fprintf(['The Error of Original ELM is ',num2str(dist_sample_el),'.\n'])
fprintf(['The Error of Robust ELM is ',num2str(dist_sample_ro),'.\n'])

fprintf(['Distance (Reachable set method) between ELM and original model is dist_reach = ',  num2str(e_max_reach_el),'.\n'])

fprintf(['Distance (Reachable set method) between Robust ELM and original model is dist_reach = ',  num2str(e_max_reach_ro),'.\n'])

