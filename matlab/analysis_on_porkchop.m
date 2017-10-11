a =load('../output/porkchop.txt');
fi = load('../output/candidates.txt');

%vel = [4.90139822e+00 -1.75336121e+00 -2.12667455e+00]
%norm(vel)

figure(1)
hold on;
scatter(a(:,1), a(:,2), 20, a(:,3));
scatter(fi(1:200,1),fi(1:200,2),'.','m');
scatter(fi(200:end-200,1),fi(200:end-200,2),'.','k');
scatter(fi(end-200:end,1),fi(end-200:end,2),'.','r');
scatter(fi(end,1),fi(end,2),'filled','c');