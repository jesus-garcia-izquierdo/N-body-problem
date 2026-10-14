function [x,y,z,vx,vy,vz,t] = Gravedad_clasica(m,x0,y0,z0,vx0,vy0,vz0)
%Le metemos x0=[x1_0
%               x2_0
%               x3_0]
% v0 utiliza la misma notacion
%
%Nos saca   x=[x1_0  x1_1   x1_2]  xcuerpo_instante
%             [x2_0  x2_1   x2_2]
% v utiliza la misma notacion



h=100000;
tf=2*31536000;
np=tf/h;
t(1)=0;
x=x0;
y=y0;
z=z0;
vx=vx0;
vy=vy0;
vz=vz0;


for n=[1:np]

t(n+1)=t(n)+h;

%Calculamos la distancia entre los dos cuerpos
%%%%%%%%%%%%%%%%%%%%%
delta_x=x(1,n)-x(2,n);%%%delta x se calcula para tener la aceleracion(distancia entre cuerpos)
delta_y=y(1,n)-y(2,n);
delta_z=z(1,n)-z(2,n);
%%%%%%%%%%%%%%%%%%%%%%

%Calculamos la aceleracion del primer cuerpo
%%%%%%%%%%%%%%%%%
[ax,ay,az]=a(m(2),delta_x,delta_y,delta_z);%%%Metemos la segunda masa y entonces la fuerza es igual
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculamos la posicion del 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1=x(1,n)+vx(1,n)*h+ax*h*h/2;
y1=y(1,n)+vy(1,n)*h+ay*h*h/2;
z1=z(1,n)+vz(1,n)*h+az*h*h/2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculamos la posicion del 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x2=x(2,n)+vx(2,n)*h-m(1)/m(2)*ax*h*h/2;%La masa es porque la F1=-F2 entonces a2=-M1/M2*A1
y2=y(2,n)+vy(2,n)*h-m(1)/m(2)*ay*h*h/2;
z2=z(2,n)+vz(2,n)*h-m(1)/m(2)*az*h*h/2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Introduzco los valores en x,y,z
%%%%%%%%%%%%%%%%%
xnueva=[x1;x2];
ynueva=[y1;y2];
znueva=[z1;z2];

x=[x xnueva];
y=[y ynueva];
z=[z znueva];
%%%%%%%%%%%%%%%%

%Calculamos la nueva distancia
%%%%%%%%%%%%%%%%%%%%%%%%%%
delta_x=x(1,n+1)-x(2,n+1);%%%delta x se calcula para tener la aceleracion nueva(distancia entre cuerpos)
delta_y=y(1,n+1)-y(2,n+1);
delta_z=z(1,n+1)-z(2,n+1);
%%%%%%%%%%%%%%%%%%%%%%%%

%Calculamos la nueva aceleracion del cuerpo 1
%%%%%%%%%%%%%%%%%
[ax1,ay1,az1]=a(m(2),delta_x,delta_y,delta_z);%%%Metemos la segunda masa y entonces la fuerza es igual
%%%%%%%%%%%%%%%%%

%Calculamos la velocidad del 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vx(1,n+1)=vx(1,n)+(ax+ax1)*h/2;
vy(1,n+1)=vy(1,n)+(ay+ay1)*h/2;
vz(1,n+1)=vz(1,n)+(az+az1)*h/2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Calculamos la velocidad del 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vx(2,n+1)=vx(2,n)-m(1)/m(2)*(ax+ax1)*h/2;
vy(2,n+1)=vy(2,n)-m(1)/m(2)*(ay+ay1)*h/2;
vz(2,n+1)=vz(2,n)-m(1)/m(2)*(az+az1)*h/2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


end

function [ax, ay, az]=a(M,delta_x,delta_y,delta_z)
G=6.67384e-11;
modulo=sqrt(delta_x*delta_x+delta_y*delta_y+delta_z*delta_z)^3;
ax=-G*M*delta_x/modulo;
ay=-G*M*delta_y/modulo;
az=-G*M*delta_z/modulo;


end
