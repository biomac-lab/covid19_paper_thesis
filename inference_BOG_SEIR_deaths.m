addpath(genpath('./npy-matlab/npy-matlab'))
addpath(genpath('./SEIR_deaths'))

%Inference for the metapopulation SEIR model
clear all, clc

num_times = 220;
%load pop %load population
pop       = readtable('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/data_matlab/pop.csv');
pop = pop.attr_population;

deaths    = table2array(readtable('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/data_matlab/deaths.csv'));
deaths = deaths(2:num_times+1,:);

incidence = table2array(readtable('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/data_matlab/incidence.csv'));
incidence = incidence(2:num_times+1,:);

% incidence = incidence(1:num_times,:);
num_times = size(incidence,1);
M = load('movement_matrix.mat');

num_loc=size(incidence,2);%number of locations

Td   = 9;%average reporting delay
a    = 1.85;%shape parameter of gamma distribution
b    = Td/a;%scale parameter of gamma distribution
rnds = ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers

% observation operator: obs=Hx
H=zeros(num_loc,7*num_loc+8);
for i=1:num_loc
    H(i,(i-1)*7+6)=1;
end

%observation operator: obs=Hx
H_hosp=zeros(num_loc,7*num_loc+8);
for i=1:num_loc
    H_hosp(i,(i-1)*7+7)=1;
end

num_times=size(incidence,1);
obs_truth=incidence';
obs_truth_hosp=deaths';

%set OEV
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(25,obs_truth(l,t)^2/100);
    end
end

%set OHEV
OHEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OHEV(l,t)=max(25,obs_truth_hosp(l,t)^2/100);
    end
end


num_ens = 300;%number of ensemble
pop0 = double(pop)*ones(1,num_ens);



[x,paramax,paramin]=initialize_SEIHR_deaths(pop0, num_ens, M, 1:size(pop,1)); %get parameter range

num_var=size(x,1);%number of state variables
%IF setting
Iter=400; % number of iterations

num_para=size(paramax,1); % number of parameters
theta=zeros(num_para,Iter+1); % mean parameters at each iteration
para_post=zeros(num_para,num_ens,num_times,Iter); %posterior parameters
sig=zeros(1,Iter); %variance shrinking parameter
alp=0.9;           %variance shrinking rate

SIG=(paramax-paramin).^2/4;%initial covariance of parameters

lambda=1.1;%inflation parameter to aviod divergence within each iteration

Sidx   = (1:6:6*num_loc)';
Eidx   = (2:6:6*num_loc)';
Isidx  = (3:6:6*num_loc)';
Iaidx  = (4:6:6*num_loc)';
obsidx = (5:6:6*num_loc)';
obsHidx = (6:6:6*num_loc)';

%start iteration for Iter round
t_init = tic;

%%
for n=1:Iter
    sig(n)=alp^(n-1);
    %generate new ensemble members using multivariate normal distribution
    Sigma=diag(sig(n)^2*SIG);
    if (n==1)
        %first guess of state space
        [x,~,~]    =  initialize_SEIHR_deaths(pop0,num_ens,M, 1:size(pop,1));
        para       = x(end-7:end,:);
        theta(:,1) = mean(para,2);%mean parameter
    else
        
        [x,~,~] = initialize_SEIHR_deaths(pop0, num_ens, M, 1:size(pop,1));
        para = mvnrnd(theta(:,n)',Sigma,num_ens)'; %generate parameters
        x(end-7:end,:) = para;
    end
    
    %correct lower/upper bounds of the parameters
    x = checkbound_ini_SEIR_deaths(x,pop0);
    
    %Begin looping through observations
    x_prior = zeros(num_var,num_ens,num_times);%prior
    x_post = zeros(num_var,num_ens,num_times);
    
    pop=pop0;
    obs_temp = zeros(num_loc,num_ens,num_times);   % records of reported cases
    obs_temp_H = zeros(num_loc,num_ens,num_times); % records of reported hospitalization
    
    for t=1:num_times
        if mod(t,50)==0
            [n,t]
        end
        %inflation
        x = mean(x,2)*ones(1,num_ens) + lambda*(x-mean(x,2)*ones(1,num_ens));
        
        x = checkbound_SEIR_deaths(x,pop);
        %integrate forward
        [x,pop] = SEIR_deaths(x,M,pop,t,pop0);
        
        obs_cnt = H*x;%new infection
        obs_cnt_H = H_hosp*x;%new infection
        
        obs_cnt = min(obs_cnt, 1000000);
        obs_cnt_H = min(obs_cnt_H, 20000);
        
        tot_cases_t = round(sum(mean(obs_cnt,2)));
        tot_cases_h = round(sum(mean(obs_cnt_H,2)));
        if mod(t,50)==0
            [n,t]
            display(strcat('Number of cases in t=', num2str(t),' is=',num2str(tot_cases_t)))
            display(strcat('Number of deaths in t=', num2str(t),' is=',num2str(tot_cases_h)))
        end
        
        %add reporting delay
        for k=1:num_ens
            for l=1:num_loc
                if obs_cnt(l,k)>0
                    rnd = datasample(rnds,obs_cnt(l,k));
                    for h=1:length(rnd)
                        if (t+rnd(h)<=num_times)
                            obs_temp(l,k,t+rnd(h)) = obs_temp(l,k,t+rnd(h)) + 1;
                        end
                    end
                end
            end
        end
        
        obs_temp_H(:,:,t) = obs_cnt_H; % No delay to deaths
        obs_ens   = obs_temp(:,:,t);%observation at t
        obs_ens_H = obs_temp_H(:,:,t);%observation at t
        
        x_prior(:,:,t)=x;%set prior
        
        % AJUST USING DEATHS
        % loop through local observations
        %loop through local observations
        for l=1:num_loc
            %Get the variance of the ensemble
            obs_var = OHEV(l,t);
            prior_var = var(obs_ens_H(l,:));
            post_var = prior_var*obs_var/(prior_var+obs_var);
            if prior_var==0%if degenerate
                post_var=1e-3;
                prior_var=1e-3;
            end
            prior_mean = mean(obs_ens_H(l,:));
            post_mean = post_var*(prior_mean/prior_var + obs_truth_hosp(l,t)/obs_var);
            %%%% Compute alpha and adjust distribution to conform to posterior moments
            alpha = (obs_var/(obs_var+prior_var)).^0.5;
            dy = post_mean + alpha*(obs_ens_H(l,:)-prior_mean)-obs_ens_H(l,:);
            %Loop over each state variable (connected to location l)
            rr=zeros(1,num_var);
            neighbors = union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
            neighbors = [neighbors;l];%add location l
            for i=1:length(neighbors)
                idx=neighbors(i);

                for j=1:7
                    A=cov( x((idx-1)*7+j,:),obs_ens_H(l,:));
                    rr((idx-1)*7+j)=A(2,1)/prior_var;
                end
            end
            
            for i=num_loc*7+1:num_loc*7+8
                A=cov(x(i,:),obs_ens_H(l,:));
                rr(i)=A(2,1)/prior_var;
            end
            
            %Get the adjusted variable
            dx = rr'*dy;
            x  = x+dx;
            %Corrections to DA produced aphysicalities
            x = checkbound_SEIR_deaths(x,pop);
            
        end
                
        
        % AJUST USING OBSERVATIONS
        % loop through local observations
        %loop through local observations
        for l=1:num_loc
            %Get the variance of the ensemble
            obs_var = OEV(l,t);
            prior_var = var(obs_ens(l,:));
            post_var = prior_var*obs_var/(prior_var+obs_var);
            if prior_var==0%if degenerate
                post_var=1e-3;
                prior_var=1e-3;
            end
            prior_mean = mean(obs_ens(l,:));
            post_mean = post_var*(prior_mean/prior_var + obs_truth(l,t)/obs_var);
            %%%% Compute alpha and adjust distribution to conform to posterior moments
            alpha = (obs_var/(obs_var+prior_var)).^0.5;
            dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);
            %Loop over each state variable (connected to location l)
            rr=zeros(1,num_var);
            neighbors = union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
            neighbors = [neighbors;l];%add location l
            
            for i=1:length(neighbors)
                idx=neighbors(i);
                for j=1:7
                    A=cov( x((idx-1)*5+j,:),obs_ens(l,:));
                    rr((idx-1)*5+j)=A(2,1)/prior_var;
                end
            end
            
            for i=num_loc*7+1:num_loc*7+8
                A=cov(x(i,:),obs_ens(l,:));
                rr(i)=A(2,1)/prior_var;
            end
            
            %Get the adjusted variable
            dx = rr'*dy;
            x  = x+dx;
            %Corrections to DA produced aphysicalities
            x = checkbound_SEIR_deaths(x,pop);
            
        end
        
        x_post(:,:,t)      = x;
        para_post(:,:,t,n) = x(end-7:end,:);
    end
    
    para         = x_post(end-7:end,:,1:num_times);
    temp         = squeeze(mean(para,2)); %average over ensemble members
    theta(:,n+1) = mean(temp,2);  %average over time
    
        % loc_mas500 = 1:2;
        %     if mod(n,100)==0 || n==1
        %         for special_loc = loc_mas500
        %             figure
        %             obs_fitted = obs_temp(special_loc,:,:);
        %             obs_fitted =  mean((obs_fitted),2);
        %             
        %             obs_fitted_h = obs_temp_H(special_loc,:,:);
        %             obs_fitted_h =  mean((obs_fitted_h),2);
        %             
        %             subplot(1,2,1)
        %             plot(1:num_times, squeeze(obs_fitted),'k-','LineWidth',2)
        %             hold on
        %             plot(1:num_times, obs_truth(special_loc,:),'rx')
        %             legend('Fit','Data')
        %             hold off
        %             
        %             subplot(1,2,2)
        %             plot(1:num_times, squeeze(obs_fitted_h),'k--','LineWidth',2)
        %             hold on
        %             plot(1:num_times, obs_truth_hosp(special_loc,:),'bx')
        %             legend('Fit deaths','Data deaths')
        %             hold off
        %             
        %         end
        %         drawnow
        %     end
    if mod(n,50)==0
        save(strcat( '/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/',num2str(n),'_xstates_BOG') , 'x_post');
        save(strcat( '/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/',num2str(n),'_parapost_states_BOG') , 'para_post');
        para_post_mean = mean(para_post(:,:,:,1:n), 4);
        save(strcat( '/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/',num2str(n),'_parapost-mean_states_col') , 'para_post_mean');
    end
    parameters = theta(:,end); % estimated parameters
end

%%%%
t_end = toc;
print('Time running: '+num2str(t_end-t_init))

%% Define confidence interval function

% x is a vector, matrix, or any numeric array of data. NaNs are ignored.
% p is the confidence level (ie, 95 for 95% CI)
% The output is 1x2 vector showing the [lower,upper] interval values.
CIFcn = @(x) prctile(x,abs([0,100]-(100-95)/2));

%% Calculate confidence intervals
obsidx=(5:5:5*num_loc)';
obs_sim = obs_temp; %x_post_sim(obsidx, :,:);

%save('xstates_forecast_col_may22_may20_t_sim_136','x_post_sim');
obs_sim = sort(obs_sim,2);

ci = zeros(2,num_loc,num_times);
for loc =1:num_loc
    for t = 1:num_times
        ens_data = squeeze(obs_sim(loc,:,t));
        ci(:,loc,t) = CIFcn(ens_data);
    end
end

loc_mas500 = 1:10;

up_ci = squeeze(ci(1,:,:));
low_ci = squeeze(ci(2,:,:));

%%
loc_mas500 = 1:5;
cit = {'Kenedy','2','3', '4', '5', '6', 'VILLAVICENCIO', 'SAN ANDRES DE TUMACO', 'MEDELLIN', 'BUENAVENTURA'};

i=1;
num_forecast= 0;
for special_loc = loc_mas500
    figure
    obs_sim_i = obs_sim(special_loc,:,:);
    
    obs_fitted =  squeeze(median((obs_sim_i),2));
    plot(1:num_times, obs_fitted(1:num_times),'k-','LineWidth',2)
    
    hold on
    plot(num_times+1:num_times+num_forecast,obs_fitted(num_times+1:num_times+num_forecast),'b-','LineWidth',2)
    
    plot(1:num_times, obs_truth(special_loc,:),'ro','LineWidth',2)
    
    plot(1:num_times, up_ci(special_loc,1:num_times),'Color', uint8([17 17 17]), 'LineStyle','--','LineWidth',2)
    plot(1:num_times, low_ci(special_loc,1:num_times),'Color', uint8([17 17 17]), 'LineStyle','--','LineWidth',2)
    
    %plot(num_times+1:num_times+num_forecast, up_ci(special_loc,num_times+1:num_times+num_forecast),'Color', 'b', 'LineStyle','--','LineWidth',2)
    %plot(num_times+1:num_times+num_forecast, low_ci(special_loc,num_times+1:num_times+num_forecast),'Color','b', 'LineStyle','--','LineWidth',2)
    
    title(cit{i})
    legend('Fit','Data')
    hold off
    i=i+1;
    
end

