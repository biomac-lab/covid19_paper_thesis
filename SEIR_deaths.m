function [x,pop] = SEIHR(x,M,pop,ts,pop0)
%the metapopulation SEIHR model
dt=1;
tmstep=1;

%integrate forward for one day
num_loc     = size(pop,1);
[~,num_ens] = size(x);


num_state_var = 7;

% S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
Sidx   = (1:num_state_var:num_state_var*num_loc)';
Eidx   = (2:num_state_var:num_state_var*num_loc)';
Isidx  = (3:num_state_var:num_state_var*num_loc)';
Iaidx  = (4:num_state_var:num_state_var*num_loc)';
Ididx  = (5:num_state_var:num_state_var*num_loc)';
obsidx = (6:num_state_var:num_state_var*num_loc)';
obsHidx = (7:num_state_var:num_state_var*num_loc)';

betaidx   = num_state_var*num_loc+1;
muidx     = num_state_var*num_loc+2;
thetaidx  = num_state_var*num_loc+3;
Zidx      = num_state_var*num_loc+4;
alphaidx  = num_state_var*num_loc+5;
Didx      = num_state_var*num_loc+6;
Lidx      = num_state_var*num_loc+7;
IFRidx    = num_state_var*num_loc+8;

S  = zeros(num_loc,num_ens,tmstep+1);
E  = zeros(num_loc,num_ens,tmstep+1);
Is = zeros(num_loc,num_ens,tmstep+1);
Ia = zeros(num_loc,num_ens,tmstep+1);

Incidence       = zeros(num_loc,num_ens,tmstep+1);
Deaths = zeros(num_loc,num_ens,tmstep+1);

obs       = zeros(num_loc,num_ens);
obs_D       = zeros(num_loc,num_ens);

%initialize S,E,Is,Ia and parameters
S(:,:,1)  = x(Sidx,:);
E(:,:,1)  = x(Eidx,:);
Is(:,:,1) = x(Isidx,:);
Ia(:,:,1) = x(Iaidx,:);
Id(:,:,1) = x(Ididx,:);

beta  = x(betaidx,:);
mu    = x(muidx,:);
theta = x(thetaidx,:);
Z     = x(Zidx,:);
alpha = x(alphaidx,:);
D     = x(Didx,:);
L     = x(Lidx,:);
IFR   = x(IFRidx,:);


%start integration
tcnt=0;
for t= ts+dt:dt:ts+tmstep
    
    tcnt=tcnt+1;
    dt1=dt;
    
    stay1 = Is(:,:,tcnt)+Id(:,:,tcnt);
    %first step
    % susceptible
    ESenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(S(:,:,tcnt)./(pop-stay1)));
    ESleft   = min(dt1*(ones(num_loc,1)*theta).*(S(:,:,tcnt)./(pop-stay1)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*S(:,:,tcnt));    % susceptible

    % Exposed
    EEenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(E(:,:,tcnt)./(pop-stay1)));
    EEleft   = min(dt1*(ones(num_loc,1)*theta).*(E(:,:,tcnt)./(pop-stay1)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*E(:,:,tcnt));

    % Asymptomatic
    EIaenter = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Ia(:,:,tcnt)./(pop-stay1)));
    EIaleft  = min(dt1*(ones(num_loc,1)*theta).*(Ia(:,:,tcnt)./(pop-stay1)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Ia(:,:,tcnt));
 
    % Symptomatic and hospitalized do not move, that's why we do not compute EIsenter and EIsleft

    % S*I/N -> E
    Eexps = dt1*(ones(num_loc,1)*beta).*S(:,:,tcnt).*Is(:,:,tcnt)./pop;    
    % S*A/N -> E
    Eexpa = dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*S(:,:,tcnt).*Ia(:,:,tcnt)./pop;
    % E -> Is
    Einfs = dt1*(ones(num_loc,1)*alpha.*(1-IFR)).*E(:,:,tcnt)./(ones(num_loc,1)*Z);
    % E -> Ia
    Einfa = dt1*(ones(num_loc,1)*(1-alpha).*(1-IFR)).*E(:,:,tcnt)./(ones(num_loc,1)*Z);    
    % E -> Id
    Einfd = dt1*(ones(num_loc,1).*IFR).*E(:,:,tcnt)./(ones(num_loc,1)*Z);
    % Id -> D 
    Edeath = dt1*(ones(num_loc,1)).*Id(:,:,tcnt)./(ones(num_loc,1)*L);
    % I -> R
    Erecs = dt1*Is(:,:,tcnt)./(ones(num_loc,1)*D);
    % A -> R
    Ereca = dt1*Ia(:,:,tcnt)./(ones(num_loc,1)*D);

    % Susceptible Enter ; Susceptible Left;
    ESenter  = max(ESenter,0);   ESleft  = max(ESleft,0);
    % Exposed enter ; Exposed Left
    EEenter  = max(EEenter,0);   EEleft  = max(EEleft,0);
    % Asymptomatic enter; Asymptomatic left; 
    EIaenter = max(EIaenter,0);  EIaleft = max(EIaleft,0);    
    
    % New infections do to (S->E):
    % Symptomatic (I)   % Asymptomatic (A)
    Eexps = max(Eexps,0); Eexpa=max(Eexpa,0); 
    
    % New infections do to (E->alpha*I ; E->(1-alpha)*A);
    % Symptomatic (E->I)   % Asymptomatic (E->A)
    Einfs = max(Einfs,0); Einfa=max(Einfa,0);   
    
    % New hospitalizations (I->H) (assume only I are hospitalized)
    Edeath = max(Edeath,0);
    Einfd = max(Einfd,0);

    % Recuperations:
    % Symptomatic (I)   % Asymptomatic (A)  % Hospitalized (H)
    Erecs=max(Erecs,0); Ereca=max(Ereca,0);
    
    %%%%%%%%%% stochastic version
    ESenter=poissrnd(ESenter);ESleft=poissrnd(ESleft);
    EEenter=poissrnd(EEenter);EEleft=poissrnd(EEleft);
    EIaenter=poissrnd(EIaenter);EIaleft=poissrnd(EIaleft);
    Eexps=poissrnd(Eexps);
    Eexpa = poissrnd(Eexpa);
    Einfs = poissrnd(Einfs);
    Einfa = poissrnd(Einfa);
    Einfd = poissrnd(Einfd);
    Edeath = poissrnd(Edeath);
    Erecs = poissrnd(Erecs);
    Ereca = poissrnd(Ereca);

    
    sk1  = -Eexps-Eexpa+ESenter-ESleft;
    ek1  = Eexps+Eexpa-Einfs-Einfa-Einfd+EEenter-EEleft;
    isk1 = Einfs-Erecs; % symptomatic do not move
    iak1 = Einfa-Ereca+EIaenter-EIaleft;
    idk1 = Einfd-Edeath;
    ik1i = Einfs;
    ik1h = Edeath;
    %%%%%%% FINISH FIRST INTEGRATION STEP %%%%%%%
    
    %second step
    Ts1  = S(:,:,tcnt)  + sk1/2;
    Te1  = E(:,:,tcnt)  + ek1/2;
    Tis1 = Is(:,:,tcnt) + isk1/2;
    Tia1 = Ia(:,:,tcnt) + iak1/2;
    Tid1 = Id(:,:,tcnt) + idk1/2;
    
    % symptomatic and hospitalized do not move
    stay2 = Tis1+Tid1; 
    
    ESenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Ts1./(pop-stay2)));
    ESleft   = min(dt1*(ones(num_loc,1)*theta).*(Ts1./(pop-stay2)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Ts1);
    EEenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Te1./(pop-stay2)));
    EEleft   = min(dt1*(ones(num_loc,1)*theta).*(Te1./(pop-stay2)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Te1);
    EIaenter = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Tia1./(pop-stay2)));
    EIaleft  = min(dt1*(ones(num_loc,1)*theta).*(Tia1./(pop-stay2)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Tia1);
    
    % S*I/N -> E
    Eexps = dt1*(ones(num_loc,1)*beta).*Ts1.*Tis1./pop;
    % S*A/N -> E
    Eexpa = dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts1.*Tia1./pop;
    % alpha*E -> I
    Einfs = dt1*(ones(num_loc,1)*alpha.*(1-IFR)).*Te1./(ones(num_loc,1)*Z);
    % (1-alpha)*E -> I
    Einfa = dt1*(ones(num_loc,1)*(1-alpha).*(1-IFR)).*Te1./(ones(num_loc,1)*Z);    
    % E -> Id    
    Einfd = dt1*(ones(num_loc,1).*IFR).*Te1./(ones(num_loc,1)*Z);    
    % Id -> D 
    Edeath = dt1*(ones(num_loc,1)).*Tid1(:,:,tcnt)./(ones(num_loc,1)*L);
    % I -> R
    Erecs = dt1*Tis1./(ones(num_loc,1)*D);
    % A -> R
    Ereca = dt1*Tia1./(ones(num_loc,1)*D);
   
    
    ESenter  = max(ESenter,0); ESleft   = max(ESleft,0);
    EEenter  = max(EEenter,0); EEleft   = max(EEleft,0);
    EIaenter = max(EIaenter,0); EIaleft = max(EIaleft,0);
    Eexps  = max(Eexps,0); Eexpa = max(Eexpa,0);
    Einfs  = max(Einfs,0); Einfa = max(Einfa,0);
    Erecs  = max(Erecs,0);  Ereca = max(Ereca,0);
    Edeath = max(Edeath,0);
    Einfd = max(Einfd,0);

    %%%%%%%%%% stochastic version
    ESenter  = poissrnd(ESenter);  ESleft  = poissrnd(ESleft);
    EEenter  = poissrnd(EEenter);  EEleft  = poissrnd(EEleft);
    EIaenter = poissrnd(EIaenter); EIaleft = poissrnd(EIaleft);
    Eexps    = poissrnd(Eexps);
    
    Eexpa  = poissrnd(Eexpa);
    Einfs  = poissrnd(Einfs);
    Einfa  = poissrnd(Einfa);
    Einfd = poissrnd(Einfd);
    Edeath = poissrnd(Edeath);
    Erecs  = poissrnd(Erecs);
    Ereca  = poissrnd(Ereca);
    
    sk2  = -Eexps-Eexpa+ESenter-ESleft;
    ek2  = Eexps+Eexpa-Einfs-Einfa-Einfd+EEenter-EEleft;
    isk2 = Einfs-Erecs; % symptomatic do not move
    iak2 = Einfa-Ereca+EIaenter-EIaleft;
    idk2 = Einfd-Edeath;
    ik2i = Einfs;
    ik2h = Edeath;
    
    %%%%%%% FINISH second INTEGRATION STEP %%%%%%%

    % third step
    Ts2  = S(:,:,tcnt)  + sk2/2;
    Te2  = E(:,:,tcnt)  + ek2/2;
    Tis2 = Is(:,:,tcnt) + isk2/2;
    Tia2 = Ia(:,:,tcnt) + iak2/2;
    Tid2 = Id(:,:,tcnt) + idk2/2;
    
   
    % symptomatic and hospitalized do not move
    stay3 = Tis2+Tid2; 
    
    ESenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Ts2./(pop-stay3)));
    ESleft   = min(dt1*(ones(num_loc,1)*theta).*(Ts2./(pop-stay3)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Ts2);
    EEenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Te2./(pop-stay3)));
    EEleft   = min(dt1*(ones(num_loc,1)*theta).*(Te2./(pop-stay3)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Te2);
    EIaenter = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Tia1./(pop-stay3)));
    EIaleft  = min(dt1*(ones(num_loc,1)*theta).*(Tia2./(pop-stay3)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Tia2);
    
    % S*I/N -> E
    Eexps = dt1*(ones(num_loc,1)*beta).*Ts2.*Tis2./pop;
    % S*A/N -> E
    Eexpa = dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts2.*Tia2./pop;
    % (1-IFR)*alpha*E -> I
    Einfs = dt1*(ones(num_loc,1)*alpha.*(1-IFR)).*Te2./(ones(num_loc,1)*Z);
    % (1-IFR)*(1-alpha)*E -> I
    Einfa = dt1*(ones(num_loc,1)*(1-alpha).*(1-IFR)).*Te2./(ones(num_loc,1)*Z);
    % IFR*E -> Id
    Einfd =  dt1*(ones(num_loc,1).*IFR).*Te2./(ones(num_loc,1)*Z); 
    % Id -> D
    Edeath =  dt1*(ones(num_loc,1)).*Tid2./(ones(num_loc,1)*L);
    % I -> R
    Erecs = dt1*Tis2./(ones(num_loc,1)*D);
    % A -> R
    Ereca = dt1*Tia2./(ones(num_loc,1)*D);
   
    ESenter=max(ESenter,0);ESleft=max(ESleft,0);
    EEenter=max(EEenter,0);EEleft=max(EEleft,0);
    EIaenter=max(EIaenter,0);EIaleft=max(EIaleft,0);
    Eexps=max(Eexps,0);Eexpa=max(Eexpa,0);
    Einfs=max(Einfs,0);Einfa=max(Einfa,0);
    Erecs=max(Erecs,0);Ereca=max(Ereca,0);
    Edeath=max(Edeath,0);
    Einfd=max(Einfd,0);

    %%%%%%%%%% stochastic version
    ESenter  = poissrnd(ESenter);   ESleft  = poissrnd(ESleft);
    EEenter  = poissrnd(EEenter);   EEleft  = poissrnd(EEleft);
    EIaenter = poissrnd(EIaenter); EIaleft = poissrnd(EIaleft);
    
    Eexps    = poissrnd(Eexps);
    Eexpa = poissrnd(Eexpa);
    Einfs = poissrnd(Einfs);
    Einfa = poissrnd(Einfa);
    Einfd = poissrnd(Einfd);
    Edeath = poissrnd(Edeath);
    Erecs = poissrnd(Erecs);
    Ereca = poissrnd(Ereca);

    sk3  = -Eexps-Eexpa+ESenter-ESleft;
    ek3  = Eexps+Eexpa-Einfs-Einfa-Einfd+EEenter-EEleft;
    isk3 = Einfs-Erecs; % symptomatic do not move
    iak3 = Einfa-Ereca+EIaenter-EIaleft;
    idk3 = Einfd-Edeath;
    ik3i = Einfs;
    ik3h = Edeath;
    
    %%%%%%% FINISH third INTEGRATION STEP %%%%%%%

    %fourth step
    Ts3  = S(:,:,tcnt)  + sk3;
    Te3  = E(:,:,tcnt)  + ek3;
    Tis3 = Is(:,:,tcnt) + isk3;
    Tia3 = Ia(:,:,tcnt) + iak3;
    Tid3 = Id(:,:,tcnt) + idk3;

    % symptomatic and hospitalized do not move
    stay3 = Tis3+Tid3; 
    
    ESenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Ts3./(pop-stay3)));
    ESleft   = min(dt1*(ones(num_loc,1)*theta).*(Ts3./(pop-stay3)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Ts3);
    EEenter  = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Te1./(pop-stay3)));
    EEleft   = min(dt1*(ones(num_loc,1)*theta).*(Te3./(pop-stay3)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Te3);
    EIaenter = dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Tia1./(pop-stay3)));
    EIaleft  = min(dt1*(ones(num_loc,1)*theta).*(Tia3./(pop-stay3)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Tia3);
    
    % S*I/N -> E
    Eexps = dt1*(ones(num_loc,1)*beta).*Ts3.*Tis3./pop;
    % S*A/N -> E
    Eexpa = dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts3.*Tia3./pop;
    % alpha*E -> Is
    Einfs = dt1*(ones(num_loc,1)*alpha.*(1-IFR)).*Te1./(ones(num_loc,1)*Z);
    % (1-alpha)*E -> Ia
    Einfa = dt1*(ones(num_loc,1)*(1-alpha).*(1-IFR)).*Te3./(ones(num_loc,1)*Z);
    % (1-alpha)*E -> Id
    Einfd = dt1*(ones(num_loc,1).*IFR).*Te3./(ones(num_loc,1)*Z);
    % Id -> d
    Edeath = dt1*(ones(num_loc,1)).*Tis3./(ones(num_loc,1)*L);
    % I -> R
    Erecs = dt1*Tis3./(ones(num_loc,1)*D);
    % A -> R
    Ereca = dt1*Tia3./(ones(num_loc,1)*D);
  
    
    ESenter=max(ESenter,0);ESleft=max(ESleft,0);
    EEenter=max(EEenter,0);EEleft=max(EEleft,0);
    EIaenter=max(EIaenter,0);EIaleft=max(EIaleft,0);
    Eexps=max(Eexps,0);Eexpa=max(Eexpa,0);
    Einfs=max(Einfs,0);Einfa=max(Einfa,0);
    Erecs=max(Erecs,0);Ereca=max(Ereca,0);
    Edeath=max(Edeath,0);
    Einfd=max(Einfd,0);

    %%%%%%%%%% stochastic version
    ESenter=poissrnd(ESenter);ESleft=poissrnd(ESleft);
    EEenter=poissrnd(EEenter);EEleft=poissrnd(EEleft);
    EIaenter=poissrnd(EIaenter);EIaleft=poissrnd(EIaleft);
    Eexps=poissrnd(Eexps);

    Eexpa = poissrnd(Eexpa);
    Einfs = poissrnd(Einfs);
    Einfa = poissrnd(Einfa);
    Einfd=poissrnd(Einfd);
    Edeath = poissrnd(Edeath);
    Erecs = poissrnd(Erecs);
    Ereca = poissrnd(Ereca);

    sk4  = -Eexps-Eexpa+ESenter-ESleft;
    ek4  = Eexps+Eexpa-Einfs-Einfa-Einfd+EEenter-EEleft;
    isk4 = Einfs-Erecs; % symptomatic do not move
    iak4 = Einfa-Ereca+EIaenter-EIaleft;
    idk4 = Einfd-Edeath;
    ik4i = Einfs;
    ik4h = Edeath;

    %%%%%%% FINISH fourth INTEGRATION STEP %%%%%%%
    
    %%%%
    
    S(:,:,tcnt+1)         = S(:,:,tcnt)  + round(sk1/6+sk2/3+sk3/3+sk4/6);
    E(:,:,tcnt+1)         = E(:,:,tcnt)  + round(ek1/6+ek2/3+ek3/3+ek4/6);
    Is(:,:,tcnt+1)        = Is(:,:,tcnt) + round(isk1/6+isk2/3+isk3/3+isk4/6);
    Ia(:,:,tcnt+1)        = Ia(:,:,tcnt) + round(iak1/6+iak2/3+iak3/3+iak4/6);
    Id(:,:,tcnt+1)        = Id(:,:,tcnt) + round(idk1/6+idk2/3+idk3/3+idk4/6);

    Incidence(:,:,tcnt+1) = round(ik1i/6 + ik2i/3 + ik3i/3 + ik4i/6);
    Deaths(:,:,tcnt+1) = round(ik1h/6 + ik2h/3 + ik3h/3 + ik4h/6);

    obs   = Incidence(:,:,tcnt+1);
    obs_D = Deaths(:,:,tcnt+1);
end

%%% update x
x(Sidx,:)   = S(:,:,tcnt+1);
x(Eidx,:)   = E(:,:,tcnt+1);
x(Isidx,:)  = Is(:,:,tcnt+1);
x(Iaidx,:)  = Ia(:,:,tcnt+1);
x(obsidx,:) = obs;
x(obsHidx,:) = obs_D;

%%% update pop
pop=pop-sum(M(:,:,ts),1)'*theta+sum(M(:,:,ts),2)*theta;
minfrac=0.6;

pop(pop<minfrac*pop0)=pop0(pop<minfrac*pop0)*minfrac;