addpath(genpath('./npy-matlab/npy-matlab'))
addpath(genpath('./SEIR_deaths'))

%Inference for the metapopulation SEIR model
clear all, clc


%load pop %load population
pop = readtable('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/data_matlab/pop.csv');
pop = double(pop.attr_population);

deaths    = table2array(readtable('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/data_matlab/deaths.csv'));
deaths = deaths(2:end,:);

incidence = table2array(readtable('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/data_matlab/incidence.csv'));
incidence = incidence(2:end,:);

% incidence = incidence(1:num_times,:);
num_times = 220;
M = zeros(size(incidence,2),size(incidence,2),num_times);
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


num_ens = 300; % number of ensemble
pop0 = double(pop)*ones(1,num_ens);


% SIG = (paramax-paramin).^2/4;%initial covariance of parameters
lambda = 1.1;%inflation parameter to aviod divergence within each iteration

%start iteration for Iter round
t_init = tic;

[x,~,~] = initialize_SEIHR_deaths(pop0,num_ens,M, size(pop,1));
num_forecast = 100;

%Begin looping through observations
num_var = size(x,1);

x_post  = load(strcat('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/', '600_xstates_BOG'));
x_post  = x_post.x_post;

para_post  = load(strcat('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/', '600_parapost-mean_states_col'));
para_post  = para_post.para_post_mean;
theta      = squeeze(mean(para_post,3));

num_times = size(x_post,3);

pop      = pop0;
x = checkbound_ini_SEIR_deaths(x,pop0);

M = zeros(num_loc,num_loc,num_times+num_forecast);

display('Starting to simulate')
para_time = para_post;

%Begin looping through observations
x_prior = zeros(num_var,num_ens,num_times+num_forecast);%prior
x_post_sim = zeros(num_var,num_ens,num_times+num_forecast);
x_post_sim_asym = zeros(num_var,num_ens,num_times+num_forecast);

pop=pop0;
obs_temp = zeros(num_loc,num_ens,num_times+num_forecast);   % records of reported cases
obs_temp_H = zeros(num_loc,num_ens,num_times+num_forecast); % records of reported hospitalization

for t=1:num_times+num_forecast
    
    if t==1 %<=num_times
        x = x_post(:,:,t);
        x_asym = x_post(:,:,t);

    end
    
    if t<=num_times
        % para_t = para_time(:,:,t);
        para_t = mean(para_time(:,:,:),3);
        paramax = squeeze(max(para_t,[],2));
        paramin = squeeze(min(para_t,[],2));
        
        SIG     = (paramax-paramin).^2/4;%initial covariance of parameters
        Sigma   = diag( SIG );
        %para   = mvnrnd(mena',Sigma,num_ens)'; %generate parameters
        para   = mvnrnd(mean(para_t,2), Sigma,num_ens)'; %generate parameters
        x(end-7:end,:) = para;
        x_asym(end-7:end,:) = para;
    else
        para_t = mean(para_time(:,:,1:num_times),3);
        para_t(1,:) = mean(para_time(1,:,num_times-7:num_times),3);
        paramax = squeeze(max(para_t,[],2));
        paramin = squeeze(min(para_t,[],2));
        
        SIG     = (paramax-paramin).^2/4;%initial covariance of parameters
        Sigma   = diag( SIG );
        para   = mvnrnd(mean(para_t,2), Sigma,num_ens)'; %generate parameters
        x(end-7:end,:) = para;
        x_asym(end-7:end,:) = para;

    end
    
    % Inflation
    x = mean(x,2)*ones(1,num_ens) + lambda*(x-mean(x,2)*ones(1,num_ens));
    x_asym = mean(x_asym,2)*ones(1,num_ens) + lambda*(x_asym-mean(x_asym,2)*ones(1,num_ens));

    x = checkbound_SEIR_deaths(x,pop);
    x_asym = checkbound_SEIR_deaths(x_asym,pop);


    Iaidx  = (4:7:7*num_loc)';
    
    % Assymptomatic set to 0
    x_asym(end-6,:) = 0;
    x_asym(Iaidx,:) = 0;
    
    %integrate forward
    [x,pop] = SEIR_deaths(x,M,pop,t,pop0);
    [x_asym,pop] = SEIR_deaths(x_asym,M,pop,t,pop0);

    
    %Corrections to DA produced aphysicalities
    x = checkbound_SEIR_deaths(x,pop);    
    x_asym = checkbound_SEIR_deaths(x_asym,pop);

    
    x_post_sim(:,:,t)      = x;
    x_post_sim_asym(:,:,t) = x_asym;

end

% where to save the forecast
save(strcat('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/', '01_sym_simulation_forecast_xstates_col') , 'x_post_sim');
save(strcat('/Users/chaosdonkey06/Dropbox/EAKF_Forecast/colombia/checkpoints_col/', '02_asym_simulation_forecast_xstates_col') , 'x_post_sim_asym');


%% Define confidence interval function
% x is a vector, matrix, or any numeric array of data. NaNs are ignored.
% p is the confidence level (ie, 95 for 95% CI)
% The output is 1x2 vector showing the [lower,upper] interval values.
conf  = 90;
CIFcn = @(x)prctile(x,abs([0,100]-(100-conf)/2));


%% Calculate confidence intervals
obsidx=(5:5:5*num_loc)';
obs_sim = x_post_sim(obsidx, :,:);

%save('xstates_forecast_col_may22_may20_t_sim_136','x_post_sim');
obs_sim = sort(obs_sim,2);

ci = zeros(2,num_loc,num_times+num_forecast);
for loc =1:num_loc
    
    for t = 1:num_times+num_forecast
        ens_data = squeeze(obs_sim(loc,:,t));
        ci(:,loc,t) = CIFcn(ens_data);
    end
end
loc_mas500 = [1:20];

up_ci = squeeze(ci(1,:,:));
low_ci = squeeze(ci(2,:,:));

%%
loc_mas500 = 1:10;
cit = {'08 - Kennedy', '07 - Bosa', '11 - Suba', '19 - Ciudad Bolívar',...
    '10 - Engativá', '09 - Fontibón', '04 - San Cristóbal','18 - Rafael Uribe Uribe',...
    '01 - Usaquén', '16 - Puente Aranda','05 - Usme', '14 - Los Mártires',...
    '02 - Chapinero', '06 - Tunjuelito','03 - Santafe', '13 - Teusaquillo',...
    '15 - Antonio Nariño','12 - Barrios Unidos', '21 - Fuera de Bogotá', '17 - La Candelaria'};

i=1;
num_forecast_plot = 7;
for special_loc = loc_mas500
    figure
    obs_sim_i = obs_sim(special_loc,:,:);
    
    obs_fitted =  squeeze(median((obs_sim_i),2));
    plot(1:num_times, obs_fitted(1:num_times),'k-','LineWidth',2)
    
    hold on
    plot(num_times+1:num_times+num_forecast_plot,obs_fitted(num_times+1:num_times+num_forecast_plot),'b-','LineWidth',2)
    
    plot(1:num_times, obs_truth(special_loc,:),'ro','LineWidth',2)
    
    plot(1:num_times, up_ci(special_loc,1:num_times),'Color', uint8([17 17 17]), 'LineStyle','--','LineWidth',2)
    plot(1:num_times, low_ci(special_loc,1:num_times),'Color', uint8([17 17 17]), 'LineStyle','--','LineWidth',2)
    
    
    plot(num_times+1:num_times+num_forecast_plot, up_ci(special_loc,num_times+1:num_times+num_forecast_plot),'Color', 'b', 'LineStyle','--','LineWidth',2)
    plot(num_times+1:num_times+num_forecast_plot, low_ci(special_loc,num_times+1:num_times+num_forecast_plot),'Color','b', 'LineStyle','--','LineWidth',2)
    
    title(cit{i})
    legend('Fit','Data')
    hold off
    i=i+1;
    
end
