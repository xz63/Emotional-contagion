onsets=5:30:180;
dur=30; n=0;clear AUAll1 AUAll2
for s=1:x.nSubj
        for c=1:6;
            if c<4
                cond=side;
                r=c;
            else
                cond=3-side;
                r=c-3;
            end
            l1=size(opendata{c,1,s},1); l2=size(opendata{c,2,s},1);
            if l1==0 | l2==0; continue;end % one data is missing
            ind0=[1:l1]*x.srate(s);
            for b=1:6
                ind=find(ind0>onsets(b) & ind0<=onsets(b)+30);
                ll=length(ind);
                if ll<545; continue;end  %545 is 18 seconds is but some time 908 for AU we want identical
                ii=n+[1:545];
                if ind(545) >l1 | ind(545) >l2 ; continue;end  % one data is shortter 
                if c<4
                AUAll1(ii,1:17)=opendata{c,1,s}(ind(1:545),:);
                AUAll2(ii,1:17)=opendata{c,2,s}(ind(1:545),:);
                else
                AUAll1(ii,1:17)=opendata{c,2,s}(ind(1:545),:);
                AUAll2(ii,1:17)=opendata{c,1,s}(ind(1:545),:);
                end
                info12(ii,1)=s;
                info12(ii,2)=c;info12(ii,3)=b;
                n=n+545;
            end
        end
    end
save AUAll2  AUAll1    AUAll2
save info12 info12
return
save AUAll  AUAll 
return
figure; ii=1:40000;
subplot(5,1,1);
plot(AUAll(ii,18)); title(' 18 Pair ID');
subplot(5,1,2);
plot(AUAll(ii,19)); title('19 side');
subplot(5,1,3);
plot(AUAll(ii,20)); title('20 run');
subplot(5,1,4);
plot(AUAll(ii,21)); title('21: 1 watch movie 2 watch face');
subplot(5,1,5);
plot(AUAll(ii,22)); title('22: block');
saveas(gcf,'info.png')