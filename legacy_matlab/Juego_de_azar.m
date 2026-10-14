%Se cierran todas las variables y figuras
clear all
close all

% Notación de vectores: 1º coord. = 1º cuerpo, 2º coord = 2º cuerpo, ...

% Script que ilustra trayectorias caóticas con condiciones iniciales dadas
% por la función rand (aleatoria)

% Se declaran las condiciones iniciales para todos los cuerpos.
% Las coordenadas iniciales de posición corresponden al afelio (punto más alejado)

tf=100*31536000;% 1 año (en segundos)
h=1000; % Salto de tiempo
m=[1e10*rand();1e10*rand();1e10*rand()]; % Vector de masas
% Vector de posiciones en eje x, y,z; respectivamente:
x0=[-1e5+2e5*rand();-1e5+2e5*rand();-1e5+2e5*rand()];
y0=[-1e5+2e5*rand();-1e5+2e5*rand();-1e5+2e5*rand()];
z0=[-1e5+2e5*rand();-1e5+2e5*rand();-1e5+2e5*rand()];


% Vector de velocidades en eje x, y,z; respectivamente:
% Conservación del momento lineal: m*v+m*v=0 =>v=-m*v/m

% Cálculo de velocidades para que no se choquen
G=6.67384e-11;

v1=sqrt(G*m(2)/sqrt(x0(1)^2+y0(2)^2));
v2=sqrt(G*m(3)/sqrt(z0(3)^2+y0(2)^2));
v3=sqrt(G*m(1)/sqrt(z0(3)^2+x0(1)^2));
v1x=v1*sin(pi/4);
v1y=v1*sin(pi/4);
v2y=v2*sin(pi/4);
v2z=v2*sin(pi/4);
v3x=v3*sin(pi/4);
v3z=v3*sin(pi/4);


vx0=[-0.0015+0.003*rand();-0.0015+0.003*rand();-0.0015+0.003*rand()];
vy0=[-0.0015+0.003*rand();-0.0015+0.003*rand();-0.0015+0.003*rand()];
vz0=[-0.0015+0.003*rand();-0.0015+0.003*rand();-0.0015+0.003*rand()];

% Se introducen en la función Gravedad_clasica_2 todas las variables:
[x,y,z,vx,vy,vz,t,px,py,pz,p,L,Ecin,Epot,Etot]=Gravedad_clasica_2(m,x0,y0,z0,vx0,vy0,vz0,tf,h);



% Gráfica de la trayectoria de todos los planetas del cuerpos
figure(1)
xlabel('x(m)');
ylabel('y(m)');
grid;
title('Trayectorias de los cuerpos')
for i=1:length(m)
plot3(x(i,:),y(i,:),z(i,:));
hold on
end
legend('Trayectoria del cuerpo 1', 'Trayectoria del cuerpo 2', 'Trayectoria del cuerpo 3');
hold off


figure(2)
subplot(2,1,1)
plot(t,p) % Gráfica del momento lineal
xlabel('t(s)');
ylabel('p(kg*m/s)');
grid;
title('Momento lineal')
legend('Momento lineal total')
subplot(2,1,2)
plot(t,p-mean(p))
title('Momento lineal menos su media')
xlabel('t(s)');
ylabel('p(kg*m/s)');
grid;

figure(3)
subplot(2,1,1)
plot(t,Ecin,'b',t,Epot,'g',t,Etot,'k') % Gráfica de la Energía cinética, potencial y mecánica
xlabel('t(s)');
ylabel('Energia(J)');
grid;
title('Energia')
legend('Energia cinetica','Energia potencial','Energia total')
subplot(2,1,2)
plot(t,Etot-mean(Etot))
title('Energia total menos su media')
xlabel('t(s)');
ylabel('E(J)');
grid;

figure(4)
subplot(2,1,1)
plot(t,L) % Gráfica de la Energía cinética, potencial y mecánica
xlabel('t(s)');
ylabel('Momento angular(kg*m^2/s)');
grid;
title('Momento angular')
subplot(2,1,2)
plot(t,L-mean(L))
title('Momento angular menos su media')
xlabel('t(s)');
ylabel('Momento angular(kg*m^2/s)');
grid;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Cambio un poco la posicion del primer cuerpo
x0=[x0(1)-1/100*x0(1);x0(2);x0(3)];

% Se introducen en la función Gravedad_clasica_2 todas las variables:
[x,y,z,vx,vy,vz,t,px,py,pz,p,L,Ecin,Epot,Etot]=Gravedad_clasica_2(m,x0,y0,z0,vx0,vy0,vz0,tf,h);



% Gráfica de la trayectoria de todos los cuerpos:
figure(5)
xlabel('x(m)');
ylabel('y(m)');
grid;
title('Trayectorias de los cuerpos')
for i=1:length(m)
plot3(x(i,:),y(i,:),z(i,:));
hold on
end
legend('Trayectoria del cuerpo 1', 'Trayectoria del cuerpo 2', 'Trayectoria del cuerpo 3');
hold off


figure(6)
subplot(2,1,1)
plot(t,p) % Gráfica del momento lineal
xlabel('t(s)');
ylabel('p(kg*m/s)');
grid;
title('Momento lineal')
legend('Momento lineal total')
subplot(2,1,2)
plot(t,p-mean(p))
title('Momento lineal menos su media')
xlabel('t(s)');
ylabel('p(kg*m/s)');
grid;

figure(7)
subplot(2,1,1)
plot(t,Ecin,'b',t,Epot,'g',t,Etot,'k') % Gráfica de la Energía cinética, potencial y mecánica
xlabel('t(s)');
ylabel('Energia(J)');
grid;
title('Energia')
legend('Energia cinetica','Energia potencial','Energia total')
subplot(2,1,2)
plot(t,Etot-mean(Etot))
title('Energia total menos su media')
xlabel('t(s)');
ylabel('E(J)');
grid;

figure(8)
subplot(2,1,1)
plot(t,L) % Gráfica de la Energía cinética, potencial y mecánica
xlabel('t(s)');
ylabel('Momento angular(kg*m^2/s)');
grid;
title('Momento angular')
subplot(2,1,2)
plot(t,L-mean(L))
title('Momento angular menos su media')
xlabel('t(s)');
ylabel('Momento angular(kg*m^2/s)');
grid;
